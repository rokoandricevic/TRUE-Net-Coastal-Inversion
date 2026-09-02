#!/usr/bin/env python3
"""Full-Domain Inference for TRUE-Net (Chronological t_mod Routing).

Executes:
- Single-Pass Forward Inference via Interpolated Input Normalization.
- Continuous 12-Channel Assembly (10 Optical + 2 Circular Temporal).
- Aleatory Uncertainty scaling via Spectral Trust Index.
- Epistemic Uncertainty via Ensemble Variance.
- Chronological Circular Blending of Seasonal Anchors.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import TRUE_Config as cfg

SCRIPT_VERSION = "1.0.0-tmod"

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_XLARGE_DIRNAME = "prepared_xlarge" 
DEFAULT_TRAINING_DIRNAME = "TRUE_Net_Training_Diagnostics"

# Updated Output Directories for TRUE-Net
DEFAULT_OUTPUT_DIRNAME = "TRUE_Net_Inference_v1_0"
MANIFEST_FILENAME = "TRUE_Net_Inference_Manifest.csv"

CHANNEL_NAMES = [
    "B02", "B03", "B04", "B05", "B06", "B07", 
    "B02/B03", "B03/B04", "B03/(B02+B04)", "NDCI"
]
EXPECTED_ANCHOR_ORDER = ("June21", "November21", "March22", "April22")


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
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        u2 = self.up2(b)
        if u2.shape != e2.shape: u2 = torch.nn.functional.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        if u1.shape != e1.shape: u1 = torch.nn.functional.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.final(d1)


def get_calendar_weights(doy: int) -> dict[str, float]:
    """Calculates linear calendar blending weights based on anchor DOY."""
    anchor_doys = {k: v["doy"] for k, v in cfg.PRIOR_CONFIG.items()}
    sorted_anchors = sorted(anchor_doys.items(), key=lambda x: x[1])
    circular_anchors = sorted_anchors + [(sorted_anchors[0][0], sorted_anchors[0][1] + 365.25)]
    
    adjusted_doy = doy if doy >= sorted_anchors[0][1] else doy + 365.25
        
    for i in range(len(circular_anchors) - 1):
        a1, doy1 = circular_anchors[i]
        a2, doy2 = circular_anchors[i+1]
        
        if doy1 <= adjusted_doy <= doy2:
            dist_total = doy2 - doy1
            if dist_total < 1e-6: return {a1: 1.0, a2: 0.0}
            
            w1 = (doy2 - adjusted_doy) / dist_total
            w2 = (adjusted_doy - doy1) / dist_total
            
            if w1 > 0.999: return {a1: 1.0, a2: 0.0}
            if w2 > 0.999: return {a2: 1.0, a1: 0.0}
            return {a1: float(w1), a2: float(w2)}
    
    return {sorted_anchors[0][0]: 1.0, sorted_anchors[1][0]: 0.0}


def load_training_statistics(training_dir: Path) -> tuple[float, float, dict]:
    """Parses the global target metrics and channel normalization statistics."""
    global_df = pd.read_csv(training_dir / "Global_Target_Stats.csv")
    y_mean = global_df["GLOBAL_Y_MEAN"].iloc[0]
    y_std = global_df["GLOBAL_Y_STD"].iloc[0]
    
    stats_df = pd.read_csv(training_dir / "Channel_Normalization_Stats.csv")
    anchor_stats = {}
    for anchor in EXPECTED_ANCHOR_ORDER:
        a_df = stats_df[stats_df["Anchor"] == anchor].sort_values("Channel")
        anchor_stats[anchor] = {
            "mean": a_df["Mean"].values.astype(np.float32),
            "std": a_df["Std"].values.astype(np.float32)
        }
    return y_mean, y_std, anchor_stats


def load_ensemble(training_dir: Path, device: torch.device) -> list[UNetTRUE]:
    ensemble = []
    for m in range(5):
        model_path = training_dir / f"member_{m}.pth"
        model = UNetTRUE().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        ensemble.append(model)
    return ensemble


def run(args: argparse.Namespace) -> None:
    project_dir = args.project_dir.expanduser().resolve()
    training_dir = project_dir / DEFAULT_TRAINING_DIRNAME
    xlarge_dir = project_dir / DEFAULT_XLARGE_DIRNAME
    output_dir = project_dir / DEFAULT_OUTPUT_DIRNAME
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scenes").mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("\n" + "="*70)
    print("   FULL-DOMAIN FCN INFERENCE — TRUE-Net v1.0 (Chronological)")
    print("="*70)

    # 1. Load Geometries and Masks
    print("-> Loading XLarge Polygon Master Constraints...")
    latitude = np.load(xlarge_dir / "xl_grid_latitude.npy", allow_pickle=False)
    longitude = np.load(xlarge_dir / "xl_grid_longitude.npy", allow_pickle=False)
    base_water_mask = np.load(xlarge_dir / "xl_water_mask.npy", allow_pickle=False).astype(bool)
    
    h, w = base_water_mask.shape

    # 2. Load Neural Configurations
    y_mean, y_std, anchor_stats = load_training_statistics(training_dir)
    ensemble = load_ensemble(training_dir, device)
    print(f"-> Ensemble Loaded. Global Y_Mean: {y_mean:.4f}, Y_Std: {y_std:.4f}")

    manifest = pd.read_csv(xlarge_dir / "xlarge_scene_manifest.csv").copy()
    output_rows = []

    for scene_number, scene in manifest.iterrows():
        token = re.sub(r"[^0-9]", "", str(scene["acquisition_date"]).strip().split('.')[0])[:8]
        dt = datetime.strptime(token, "%Y%m%d")
        doy = dt.timetuple().tm_yday
        
        weights = get_calendar_weights(doy)
        weight_str = " + ".join([f"{w*100:.0f}% {a}" for a, w in weights.items() if w > 0])
        print(f"\n[{scene_number + 1:02d}/{len(manifest):02d}] Scene {token} (DOY {doy}) -> {weight_str}")

        # 3. Load Optical Data and Mask Invalid
        tensor_path = xlarge_dir / str(scene["tensor_file"])
        optical = np.load(tensor_path, allow_pickle=False)[0].astype(np.float32)
        
        # Exact optical safety: Master mask AND finite/positive satellite returns
        scene_water_mask = base_water_mask & np.isfinite(optical[..., 0]) & (optical[..., 0] > 0.0)

        # 4. Interpolate Normalization Space
        x_mean_blend = np.zeros(10, dtype=np.float32)
        x_std_blend = np.zeros(10, dtype=np.float32)
        tau_blend = 0.0
        
        for a, weight in weights.items():
            if weight > 0:
                x_mean_blend += weight * anchor_stats[a]["mean"]
                x_std_blend += weight * anchor_stats[a]["std"]
                tau_blend += weight * cfg.PRIOR_CONFIG[a]["tau_a"]
        
        # 5. Dynamic 12-Channel Assembly
        normalized = (optical - x_mean_blend) / (x_std_blend + 1e-8)
        
        t_sin_val, t_cos_val = cfg.get_temporal_modulation(doy)
        t_sin = np.full((h, w), t_sin_val, dtype=np.float32)
        t_cos = np.full((h, w), t_cos_val, dtype=np.float32)
        
        # EXPLICIT float32 cast to prevent Apple MPS crash
        x_12 = np.concatenate([normalized, t_sin[..., None], t_cos[..., None]], axis=-1).astype(np.float32)
        in_t = torch.tensor(x_12).permute(2, 0, 1).unsqueeze(0).to(device)

        # 6. Ensemble Inference
        member_preds = []
        with torch.inference_mode():
            for model in ensemble:
                pred_norm = model(in_t).squeeze().cpu().numpy()
                pred_phys = (pred_norm * y_std) + y_mean
                pred_phys = np.clip(pred_phys, 0.01, None) 
                member_preds.append(pred_phys)

        member_stack = np.stack(member_preds, axis=0)
        predictive_mean = np.mean(member_stack, axis=0)
        epistemic_var = np.var(member_stack, axis=0, ddof=0)

        # 7. Trust Diagnostic & Aleatory Uncertainty (σ_spec)
        b2b3 = optical[:, :, 6]
        b2b3_mean_blend = x_mean_blend[6]
        b2b3_std_blend = x_std_blend[6]
        
        z_map = np.zeros_like(b2b3)
        if np.sum(scene_water_mask) > 0:
            z_map[scene_water_mask] = (b2b3[scene_water_mask] - b2b3_mean_blend) / (b2b3_std_blend + 1e-8)
            
        a_spec = cfg.calculate_spectral_trust(z_map)
        
        # Corrected mathematically bounded formula for the aleatory proxy
        aleatory_sd = tau_blend * np.sqrt(1.0 / a_spec)
        
        epistemic_sd = np.sqrt(np.maximum(epistemic_var, 0.0))

        # Format arrays for output
        out_mean = np.where(scene_water_mask, predictive_mean, np.nan)
        out_aleatory = np.where(scene_water_mask, aleatory_sd, np.nan)
        out_epistemic = np.where(scene_water_mask, epistemic_sd, np.nan)
        out_trust = np.where(scene_water_mask, a_spec, np.nan)

        # 8. Save NPZ Archive
        scene_output = output_dir / "scenes" / f"TRUENet_{token}.npz"
        np.savez_compressed(
            scene_output, 
            acquisition_date=np.asarray(token), 
            latitude=latitude, 
            longitude=longitude,
            valid_water_mask=scene_water_mask,
            predictive_mean_chl=out_mean,
            aleatory_proxy_sd_chl=out_aleatory,
            epistemic_proxy_sd_chl=out_epistemic,
            spectral_trust_index=out_trust
        )

        row: dict[str, object] = {
            "acquisition_date": token,
            "assigned_regime": weight_str, 
            "valid_water_pixels": int(scene_water_mask.sum()),
            "mean_chl": float(np.nanmean(out_mean)),
            "mean_aleatory_sd": float(np.nanmean(out_aleatory)),
            "mean_epistemic_sd": float(np.nanmean(out_epistemic)),
            "mean_trust_index": float(np.nanmean(out_trust))
        }
        output_rows.append(row)

        # 9. Adaptive Visual Verification (QuickLooks)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        valid_vals = out_mean[scene_water_mask]
        vmin_dyn = float(np.nanpercentile(valid_vals, 2)) if len(valid_vals) > 0 else 0.0
        vmax_dyn = float(np.nanpercentile(valid_vals, 98)) if len(valid_vals) > 0 else 3.0
        
        if vmax_dyn - vmin_dyn < 0.25:
            mid = (vmax_dyn + vmin_dyn) / 2.0
            vmin_dyn = max(0.0, mid - 0.15)
            vmax_dyn = mid + 0.15
        
        im0 = axes[0].imshow(out_mean, cmap='turbo', vmin=vmin_dyn, vmax=vmax_dyn)
        axes[0].set_title(f"Predictive Mean Chl-a\n{token} (DOY {doy}) [{vmin_dyn:.2f} - {vmax_dyn:.2f}]")
        fig.colorbar(im0, ax=axes[0], orientation='horizontal', pad=0.1, label="Chl-a [mg/m³]")

        im1 = axes[1].imshow(out_aleatory, cmap='magma', vmin=0.0)
        axes[1].set_title(f"Aleatory Uncertainty (\u03c3_spec)\nInterpolated \u03c4 = {tau_blend:.3f}")
        fig.colorbar(im1, ax=axes[1], orientation='horizontal', pad=0.1, label="\u03c3_spec [mg/m³]")

        im2 = axes[2].imshow(out_trust, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[2].set_title(f"Spectral Trust Diagnostic (A_spec)")
        fig.colorbar(im2, ax=axes[2], orientation='horizontal', pad=0.1, label="A_spec [-]")

        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "scenes" / f"QuickLook_TRUENet_{token}.png", dpi=200)
        plt.close()

    pd.DataFrame(output_rows).to_csv(output_dir / MANIFEST_FILENAME, index=False)
    print("\nSUCCESS: All 22 scenes processed and exported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    run(parser.parse_args())