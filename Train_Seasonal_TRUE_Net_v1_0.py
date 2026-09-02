#!/usr/bin/env python3
"""Prior-Informed Spatio-Temporal TRUE-Net Training.

Executes:
- 12-Channel Input Assembly (10 Optical + 2 Circular Temporal).
- Normalized Loss Space Optimization (Base MSE + Prior Penalty).
- Automated Land Masking and Spectral Trust Evaluation.
- Comprehensive Diagnostic Generation (Loss, Compliance, Spatial, Masks).
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import TRUE_Config as cfg

DEFAULT_DATA_DIR = "prepared_data_58x20_v3"
DEFAULT_OUTPUT_DIR = "TRUE_Net_Training_Diagnostics"
LAMBDA_PRIOR = 0.1
ENSEMBLE_MEMBERS = 5
EPOCHS = 100
BATCH_SIZE = 128

ANCHOR_NAMES = ("June21", "November21", "March22", "April22")


class UNetTRUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        
        u2 = self.up2(b)
        if u2.shape != e2.shape: 
            u2 = torch.nn.functional.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        if u1.shape != e1.shape: 
            u1 = torch.nn.functional.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        return self.final(d1)


def set_deterministic_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_normalization_stats(inputs, outputs, anchor_labels, output_dir):
    print("  -> Computing channel normalization statistics...")
    valid_y = outputs[outputs > 0]
    y_mean, y_std = float(np.mean(valid_y)), float(np.std(valid_y))
    
    anchor_stats = {}
    stats_log = []
    
    for anchor in ANCHOR_NAMES:
        idxs = [i for i, a in enumerate(anchor_labels) if a == anchor]
        sub_in = inputs[idxs]
        a_mean, a_std = np.zeros(10), np.zeros(10)
        for ch in range(10):
            valid_ch = sub_in[..., ch][sub_in[..., ch] > 0]
            if len(valid_ch) > 0:
                a_mean[ch], a_std[ch] = np.mean(valid_ch), np.std(valid_ch)
            stats_log.append({
                "Anchor": anchor, "Channel": ch, 
                "Mean": a_mean[ch], "Std": a_std[ch]
            })
        anchor_stats[anchor] = {"mean": a_mean, "std": a_std}
        
    pd.DataFrame(stats_log).to_csv(output_dir / "Channel_Normalization_Stats.csv", index=False)
    pd.DataFrame([{"GLOBAL_Y_MEAN": y_mean, "GLOBAL_Y_STD": y_std}]).to_csv(
        output_dir / "Global_Target_Stats.csv", index=False)
        
    print(f"  -> Global Target Mean: {y_mean:.4f} | Std: {y_std:.4f}")
    return y_mean, y_std, anchor_stats


def build_datasets(inputs, outputs, anchor_labels, anchor_doys, real_nums, anchor_stats, y_mean, y_std, device):
    print("  -> Assembling 12-channel input tensors and applying land masks...")
    n_samples, h, w, _ = inputs.shape
    
    all_x = np.zeros((n_samples, 12, h, w), dtype=np.float32)
    all_y_true = np.zeros((n_samples, 1, h, w), dtype=np.float32)
    all_y_prior = np.zeros((n_samples, 1, h, w), dtype=np.float32)
    all_masks = np.zeros((n_samples, 1, h, w), dtype=bool)
    all_trust = np.zeros((n_samples, 1, h, w), dtype=np.float32)
    raw_b2b3 = np.zeros((n_samples, 1, h, w), dtype=np.float32)

    for i in range(n_samples):
        anchor = anchor_labels[i]
        doy = anchor_doys[i]
        
        x_raw = inputs[i]
        mask = np.isfinite(x_raw[..., 0]) & (x_raw[..., 0] > 0.0)
        
        x_norm = (x_raw - anchor_stats[anchor]["mean"]) / (anchor_stats[anchor]["std"] + 1e-8)
        
        t_sin_val, t_cos_val = cfg.get_temporal_modulation(doy)
        t_sin = np.full((h, w), t_sin_val, dtype=np.float32)
        t_cos = np.full((h, w), t_cos_val, dtype=np.float32)
        
        x_12 = np.concatenate([x_norm, t_sin[..., None], t_cos[..., None]], axis=-1)
        all_x[i] = np.transpose(x_12, (2, 0, 1))
        
        b2b3 = x_raw[..., 6]
        raw_b2b3[i, 0] = b2b3
        
        cfg_anchor = cfg.PRIOR_CONFIG[anchor]
        y_phys_raw = cfg_anchor["a"] * np.exp(cfg_anchor["b"] * b2b3)
        
        z_map = np.zeros_like(b2b3)
        if np.sum(mask) > 0:
            z_map[mask] = (b2b3[mask] - np.mean(b2b3[mask])) / (np.std(b2b3[mask]) + 1e-8)
        a_spec = cfg.calculate_spectral_trust(z_map)
        
        all_y_true[i, 0] = (outputs[i] - y_mean) / (y_std + 1e-8)
        all_y_prior[i, 0] = (y_phys_raw - y_mean) / (y_std + 1e-8)
        all_masks[i, 0] = mask
        all_trust[i, 0] = a_spec

    print("  -> Pushing tensors to computation device...")
    t_x = torch.tensor(all_x, device=device)
    t_true = torch.tensor(all_y_true, device=device)
    t_prior = torch.tensor(all_y_prior, device=device)
    t_mask = torch.tensor(all_masks, device=device)
    t_trust = torch.tensor(all_trust, device=device)
    t_doys = torch.tensor(anchor_doys, device=device)
    t_raw_b2b3 = torch.tensor(raw_b2b3, device=device)
    t_real_nums = torch.tensor(real_nums, dtype=torch.int32, device=device)

    dataset = TensorDataset(t_x, t_true, t_prior, t_mask, t_trust, t_doys, t_raw_b2b3, t_real_nums)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def plot_diagnostics(logs, loader, device, y_mean, y_std, output_dir):
    print("\n=== Generating Diagnostics ===")
    output_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(logs)
    
    print("  -> Plotting Ensemble Loss Curves...")
    plt.figure(figsize=(10, 5))
    for m in range(ENSEMBLE_MEMBERS):
        m_log = df[df['member'] == m]
        plt.plot(m_log['epoch'], m_log['mse'], color='tab:blue', alpha=0.5)
        plt.plot(m_log['epoch'], m_log['prior'], color='tab:red', alpha=0.5)
    plt.title("Ensemble Loss Convergence")
    plt.grid(True, linestyle='--')
    plt.savefig(output_dir / "Loss_Curves.png", dpi=200)
    plt.close()

    print("  -> Reconstructing Spatial Matrices...")
    model = UNetTRUE().to(device)
    model.load_state_dict(torch.load(output_dir / "member_0.pth", map_location=device, weights_only=True))
    model.eval()

    anchor_data = {a: {"b2b3": [], "pred": [], "true": []} for a in ANCHOR_NAMES}
    spatial_samples = {a: None for a in ANCHOR_NAMES}

    with torch.no_grad():
        for x, y_t, y_p, mask, trust, doys, raw_b2b3, real_nums in loader:
            pred_norm = model(x)
            pred_phys = (pred_norm * y_std) + y_mean
            true_phys = (y_t * y_std) + y_mean
            
            for i in range(len(doys)):
                doy = int(doys[i].item())
                r_num = int(real_nums[i].item())
                anchor = next((a for a, c in cfg.PRIOR_CONFIG.items() if c["doy"] == doy), None)
                if anchor:
                    m = mask[i, 0].cpu().bool().numpy()
                    anchor_data[anchor]["pred"].extend(pred_phys[i, 0].cpu().numpy()[m])
                    anchor_data[anchor]["true"].extend(true_phys[i, 0].cpu().numpy()[m])
                    anchor_data[anchor]["b2b3"].extend(raw_b2b3[i, 0].cpu().numpy()[m])
                    
                    if spatial_samples[anchor] is None:
                        spatial_samples[anchor] = {
                            "true": true_phys[i, 0].cpu().numpy(),
                            "pred": pred_phys[i, 0].cpu().numpy(),
                            "mask": m,
                            "trust": trust[i, 0].cpu().numpy(),
                            "real_num": r_num
                        }

    print("  -> Plotting Boundary Compliance...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, anchor in zip(axes.ravel(), ANCHOR_NAMES):
        b2b3 = np.array(anchor_data[anchor]["b2b3"])
        preds = np.array(anchor_data[anchor]["pred"])
        
        ax.scatter(b2b3, preds, s=2, alpha=0.3, c='royalblue', label="Predictions")
        
        if len(b2b3) > 0:
            x_vals = np.linspace(b2b3.min(), b2b3.max(), 100)
            a = cfg.PRIOR_CONFIG[anchor]["a"]
            b = cfg.PRIOR_CONFIG[anchor]["b"]
            tau = cfg.PRIOR_CONFIG[anchor]["tau_a"]
            
            y_trend = a * np.exp(b * x_vals)
            ax.plot(x_vals, y_trend, 'k--', lw=2, label="Mean Trend")
            ax.plot(x_vals, y_trend + 3*tau, 'r-', lw=1.5, label="Upper Bound")
            ax.plot(x_vals, np.maximum(0, y_trend - 3*tau), 'r-', lw=1.5, label="Lower Bound")

        ax.set_title(f"{anchor} Boundary Compliance")
        ax.set_ylabel("Chlorophyll-a [mg/m³]")
        ax.set_xlabel("Optical Ratio (B02/B03)")
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "Boundary_Compliance.png", dpi=200)
    plt.close()

    print("  -> Plotting Spatial Grids...")
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for row, anchor in enumerate(ANCHOR_NAMES):
        s = spatial_samples[anchor]
        v_true = np.where(s["mask"], s["true"], np.nan)
        v_pred = np.where(s["mask"], s["pred"], np.nan)
        v_err = np.where(s["mask"], np.abs(v_true - v_pred), np.nan)
        
        vmin, vmax = np.nanmin([v_true, v_pred]), np.nanmax([v_true, v_pred])
        
        im0 = axes[row, 0].imshow(v_true, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"{anchor} Target (Realization #{s['real_num']})")
        fig.colorbar(im0, ax=axes[row, 0], orientation='horizontal', pad=0.15, fraction=0.046, label="Chl-a [mg/m³]")
        
        im1 = axes[row, 1].imshow(v_pred, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row, 1].set_title("TRUE-Net Prediction")
        fig.colorbar(im1, ax=axes[row, 1], orientation='horizontal', pad=0.15, fraction=0.046, label="Chl-a [mg/m³]")
        
        im2 = axes[row, 2].imshow(v_err, cmap='magma')
        axes[row, 2].set_title("Absolute Error")
        fig.colorbar(im2, ax=axes[row, 2], orientation='horizontal', pad=0.15, fraction=0.046, label="Error [mg/m³]")
        
    plt.tight_layout()
    plt.savefig(output_dir / "Spatial_Grids.png", dpi=200)
    plt.close()

    print("  -> Plotting Masks and Trust Regions...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for col, anchor in enumerate(ANCHOR_NAMES):
        s = spatial_samples[anchor]
        water_px = int(np.sum(s["mask"]))
        
        axes[0, col].imshow(s["mask"].astype(float), cmap='Blues_r', vmin=0, vmax=1)
        axes[0, col].set_title(f"{anchor} Hard Mask\n{water_px} pixels")
        axes[0, col].axis('off')
        
        trust_map = np.where(s["mask"], s["trust"], np.nan)
        im = axes[1, col].imshow(trust_map, cmap='viridis', vmin=0, vmax=1)
        axes[1, col].set_title("Spectral Trust Diagnostic")
        axes[1, col].axis('off')
        fig.colorbar(im, ax=axes[1, col], orientation='vertical', fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.savefig(output_dir / "Masks_and_Trust.png", dpi=200)
    plt.close()

    print("  -> Plotting True vs Predicted Chl-a panels...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, anchor in zip(axes.ravel(), ANCHOR_NAMES):
        t_vals = np.array(anchor_data[anchor]["true"])
        p_vals = np.array(anchor_data[anchor]["pred"])
        
        ss_res = np.sum((t_vals - p_vals) ** 2)
        ss_tot = np.sum((t_vals - np.mean(t_vals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((p_vals - t_vals)**2))
        bias = np.mean(p_vals - t_vals)
        
        ax.scatter(t_vals, p_vals, alpha=0.1, color='indigo', s=2)
        
        if len(t_vals) > 0:
            min_val = min(t_vals.min(), p_vals.min())
            max_val = max(t_vals.max(), p_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='1:1 Line')
        
        textstr = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nBias = {bias:.3f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f"{anchor} Prediction vs. Ground Truth")
        ax.set_xlabel("True Chl-a [mg/m³]")
        ax.set_ylabel("Predicted Chl-a [mg/m³]")
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "True_vs_Pred_by_Anchor.png", dpi=300)
    plt.close()
    print("=== Diagnostics Complete ===")


def run(args):
    print("\n" + "="*50)
    print("   TRUE-Net SPATIO-TEMPORAL TRAINING   ")
    print("="*50)
    project_dir = args.project_dir.resolve()
    data_dir = project_dir / DEFAULT_DATA_DIR
    out_dir = project_dir / DEFAULT_OUTPUT_DIR
    out_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device Initialized: {device}")

    print("\n=== Loading Data ===")
    manifest = pd.read_csv(data_dir / "sample_manifest.csv")
    
    n_samples, h, w = len(manifest), 20, 58
    
    anchor_labels = [ANCHOR_NAMES[i] for i in manifest["anchor_index"]]
    anchor_doys = manifest["anchor_day_of_year"].values
    real_nums = manifest["realization_number"].values

    inputs = np.fromfile(data_dir / "inputs.bin", dtype=np.float32).reshape(n_samples, h, w, 10)
    outputs = np.fromfile(data_dir / "outputs.bin", dtype=np.float32).reshape(n_samples, h, w)
    print(f"  -> Successfully loaded {n_samples} samples.")

    y_mean, y_std, anchor_stats = compute_normalization_stats(inputs, outputs, anchor_labels, out_dir)
    loader = build_datasets(inputs, outputs, anchor_labels, anchor_doys, real_nums, anchor_stats, y_mean, y_std, device)

    logs = []
    print("\n=== Commencing Ensemble Training ===")
    for m in range(ENSEMBLE_MEMBERS):
        print(f"\n--- Initializing Ensemble Member {m+1}/{ENSEMBLE_MEMBERS} ---")
        set_deterministic_seed(42 + m)
        model = UNetTRUE().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        mse_crit = nn.MSELoss(reduction="none")
        
        for ep in range(1, EPOCHS + 1):
            model.train()
            tot_mse, tot_prior, px_count = 0.0, 0.0, 0
            
            for x, y_t, y_p, mask, _, _, _, _ in loader:
                optimizer.zero_grad()
                pred_norm = model(x)
                
                loss_data = torch.sum(mse_crit(pred_norm, y_t) * mask) / torch.clamp(torch.sum(mask), min=1.0)
                loss_prior = torch.sum(mse_crit(pred_norm, y_p) * mask) / torch.clamp(torch.sum(mask), min=1.0)
                
                (loss_data + (LAMBDA_PRIOR * loss_prior)).backward()
                optimizer.step()
                
                px = int(torch.sum(mask).item())
                tot_mse += loss_data.item() * px
                tot_prior += loss_prior.item() * px
                px_count += px
            
            epoch_mse = tot_mse / px_count
            epoch_prior = tot_prior / px_count
            logs.append({"member": m, "epoch": ep, "mse": epoch_mse, "prior": epoch_prior})
            
            if ep % 10 == 0 or ep == 1:
                print(f"  Epoch {ep:03d}/{EPOCHS} | Base MSE: {epoch_mse:.4f} | Prior MSE: {epoch_prior:.4f}")
        
        print(f"  -> Saving weights for Member {m}")
        torch.save(model.state_dict(), out_dir / f"member_{m}.pth")

    pd.DataFrame(logs).to_csv(out_dir / "Ensemble_Training_Loss_Log.csv", index=False)
    
    plot_diagnostics(logs, loader, device, y_mean, y_std, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    run(parser.parse_args())