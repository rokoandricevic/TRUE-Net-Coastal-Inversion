#!/usr/bin/env python3
"""Create publication graphics from TRUE-Net v1.0.

Generates:
1. Individual 4-panel scene maps (Fixed SD scaling: 0.0 - 0.4).
2. Full-basin 22-scene Atlases (Standard colorbars).
3. Lognormal Probability Density Distributions.
4. Time-Series 95% Spatial Envelopes (Domain, Regional, Combined - Scaled 0.0-1.6).
5. Uncertainty Partition Grouped Bar Chart (Scaled 0.0-0.6).
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd

try:
    import contextily as ctx
except ImportError:
    ctx = None

SCRIPT_VERSION = "1.0.0"

DEFAULT_INFERENCE_DIRNAME = "TRUE_Net_Inference_v1_0"
DEFAULT_XLARGE_DIRNAME = "prepared_xlarge" 
DEFAULT_OUTPUT_DIRNAME = "TRUE_Net_Publication_Figures_v1_0"
DEFAULT_MANIFEST = "TRUE_Net_Inference_Manifest.csv"

COLORS = {
    "Domain Mean": "#388E3C",  "Domain SD": "#7B1FA2",
    "Jadro Estuary": "#E67E22", "Kastela Bay (Rest of Bay)": "#C0392B", "Split-Brac Canal": "#2980B9",
    "Aleatory": "#E91E63", "Epistemic": "#03A9F4", "TotalSD": "#9C27B0"
}

@dataclass(frozen=True)
class Scene:
    index: int; date_token: str; date: datetime; npz_path: Path

@dataclass
class BasemapManager:
    enabled: bool; zoom: int
    def add(self, axis: plt.Axes) -> None:
        if not self.enabled or ctx is None: return
        try:
            ctx.add_basemap(axis, source=ctx.providers.Esri.WorldImagery, crs="EPSG:4326", zoom=self.zoom, reset_extent=False, attribution=False, zorder=0)
        except Exception: pass

def parse_date_token(value: object) -> str:
    text = re.sub(r"[^0-9]", "", str(value).strip().split('.')[0])
    datetime.strptime(text[:8], "%Y%m%d")
    return text[:8]

def iso_date(date_token: str) -> str:
    return datetime.strptime(date_token, "%Y%m%d").strftime("%Y-%m-%d")

def read_polygon(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, header=None, comment="#")
    numeric = frame.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    first, second = numeric.iloc[:, 0].to_numpy(np.float64), numeric.iloc[:, 1].to_numpy(np.float64)
    if np.all((first >= 42.0) & (first <= 45.0)):
        latitude, longitude = first, second
    else:
        longitude, latitude = first, second
    if not (longitude[0] == longitude[-1] and latitude[0] == latitude[-1]):
        longitude, latitude = np.append(longitude, longitude[0]), np.append(latitude, latitude[0])
    return longitude, latitude

def polygon_mask(longitude: np.ndarray, latitude: np.ndarray, pol_lon: np.ndarray, pol_lat: np.ndarray) -> np.ndarray:
    mesh_lon, mesh_lat = np.meshgrid(longitude, latitude)
    points = np.column_stack((mesh_lon.ravel(), mesh_lat.ravel()))
    return MplPath(np.column_stack((pol_lon, pol_lat)), closed=True).contains_points(points, radius=1.0e-12).reshape(mesh_lon.shape)

def create_region_masks(lon: np.ndarray, lat: np.ndarray, base_water: np.ndarray, jadro_pol: tuple, kastela_pol: tuple) -> dict[str, np.ndarray]:
    jadro_inside = polygon_mask(lon, lat, *jadro_pol)
    kastela_inside = polygon_mask(lon, lat, *kastela_pol)
    return {
        "Entire domain": base_water.copy(),
        "Jadro Estuary": base_water & jadro_inside,
        "Kastela Bay (Rest of Bay)": base_water & kastela_inside & ~jadro_inside,
        "Split-Brac Canal": base_water & ~kastela_inside & ~jadro_inside,
    }

def load_manifest(inference_dir: Path, manifest_path: Path) -> list[Scene]:
    frame = pd.read_csv(manifest_path, dtype={"acquisition_date": str})
    scenes: list[Scene] = []
    for idx, row in frame.iterrows():
        date_token = parse_date_token(row["acquisition_date"])
        npz_path = Path(f"TRUENet_{date_token}.npz")
        if not npz_path.is_absolute(): npz_path = inference_dir / "scenes" / npz_path
        scenes.append(Scene(idx, date_token, datetime.strptime(date_token, "%Y%m%d"), npz_path))
    return sorted(scenes, key=lambda s: s.date)

def add_north_arrow(axis: plt.Axes) -> None:
    axis.annotate(
        "N", xy=(0.94, 0.93), xytext=(0.94, 0.79), xycoords="axes fraction", ha="center", va="center", fontsize=9, fontweight="bold", color="white",
        arrowprops=dict(facecolor="white", edgecolor="black", width=2.3, headwidth=7), zorder=8
    )

def add_scale_bar(axis: plt.Axes, latitude_reference: float, length_km: float = 10.0) -> None:
    x_min, x_max, y_min, y_max = *axis.get_xlim(), *axis.get_ylim()
    longitude_length = length_km / (111.32 * math.cos(math.radians(latitude_reference)))
    x0, y0 = x_min + 0.055 * (x_max - x_min), y_min + 0.065 * (y_max - y_min)
    axis.plot([x0, x0 + longitude_length], [y0, y0], color="white", linewidth=4.0, solid_capstyle="butt", zorder=8)
    axis.plot([x0, x0 + longitude_length], [y0, y0], color="black", linewidth=1.5, solid_capstyle="butt", zorder=9)
    axis.text(
        x0 + longitude_length / 2.0, y0 + 0.015 * (y_max - y_min), f"{length_km:g} km", ha="center", va="bottom", fontsize=7.5, color="white",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.2), zorder=9
    )

def plot_map_layer(
    axis: plt.Axes, values: np.ndarray, valid_mask: np.ndarray, base_water: np.ndarray,
    lon: np.ndarray, lat: np.ndarray, cmap_name: str, vmin: float, vmax: float,
    title: str, basemap: BasemapManager, show_coords: bool, add_furnishings: bool = False
):
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    aspect = 1.0 / math.cos(math.radians(float(lat.mean())))
    axis.set_facecolor("#e6e2d8")
    axis.set_xlim(extent[0], extent[1]); axis.set_ylim(extent[2], extent[3])
    basemap.add(axis)
    
    display = np.asarray(values, dtype=np.float32).copy()
    display[~valid_mask] = np.nan
    cmap = plt.colormaps[cmap_name].copy(); cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    image = axis.imshow(display, extent=extent, origin="upper", aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", zorder=3)
    
    no_data = base_water & ~valid_mask
    if np.any(no_data):
        no_data_cmap = ListedColormap([(0.83, 0.83, 0.83, 0.92)])
        no_data_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
        axis.imshow(np.where(no_data, 1.0, np.nan), extent=extent, origin="upper", aspect=aspect, cmap=no_data_cmap, vmin=0.0, vmax=1.0, interpolation="nearest", zorder=4)
        
    axis.set_title(title, fontsize=11, pad=5, fontweight="semibold")
    if show_coords:
        axis.set_xlabel("Longitude [°E]", fontsize=8.5); axis.set_ylabel("Latitude [°N]", fontsize=8.5)
        axis.tick_params(labelsize=7.5)
    else:
        axis.set_xticks([]); axis.set_yticks([])
        
    if add_furnishings:
        add_north_arrow(axis); add_scale_bar(axis, float(lat.mean()))
    return image

def create_scene_4panel(
    scene: Scene, output_path: Path, bundle: np.lib.npyio.NpzFile, lon: np.ndarray, lat: np.ndarray, 
    base_water: np.ndarray, chl_vmax: float, basemap: BasemapManager, dpi: int
) -> None:
    mask = np.asarray(bundle["valid_water_mask"], dtype=bool)
    
    aleatory = bundle["aleatory_proxy_sd_chl"]
    epistemic = bundle["epistemic_proxy_sd_chl"]
    total_sd = np.sqrt(np.square(np.where(np.isnan(aleatory), 0, aleatory)) + np.square(np.where(np.isnan(epistemic), 0, epistemic)))
    total_sd = np.where(mask, total_sd, np.nan)

    # Fixed SD Scale
    sd_vmax = 0.40

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5), constrained_layout=True)
    specs = [
        (bundle["predictive_mean_chl"], "turbo", 0.0, chl_vmax, "Predictive Mean Chl-a [mg/m³]"),
        (total_sd, "turbo", 0.0, sd_vmax, "Total Predictive SD [mg/m³]"),
        (aleatory, "turbo", 0.0, sd_vmax, r"Aleatory Uncertainty ($\sigma_{spec}$) [mg/m³]"),
        (epistemic, "turbo", 0.0, sd_vmax, "Epistemic Uncertainty [mg/m³]")
    ]
    for idx, (axis, (data_arr, cmap, vmin, vmax, title)) in enumerate(zip(axes.ravel(), specs)):
        img = plot_map_layer(axis, data_arr, mask, base_water, lon, lat, cmap, vmin, vmax, title, basemap, show_coords=True, add_furnishings=(idx == 0))
        cbar = fig.colorbar(img, ax=axis, shrink=0.82, pad=0.02, extend="neither")
        cbar.ax.tick_params(labelsize=8)

    fig.legend(handles=[Patch(facecolor="#d4d4d4", edgecolor="none", label="SCL/no-data")], loc="lower center", frameon=False, fontsize=8.5)
    
    month_name = datetime.strptime(scene.date_token, "%Y%m%d").strftime("%B")
    fig.suptitle(f"{month_name} {scene.date_token} Mean Chl-a and Uncertainties", fontsize=15.5, fontweight="bold")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def create_atlas(scenes, bundle_key, lon, lat, base_water, vmin, vmax, cmap, title, output_path, dpi):
    cols, rows = 4, math.ceil(len(scenes) / 4)
    fig, axes = plt.subplots(rows, cols, figsize=(15.2, 2.7 * rows), constrained_layout=True)
    flat_axes = np.atleast_1d(axes).ravel()
    image = None
    for axis, scene in zip(flat_axes, scenes):
        with np.load(scene.npz_path, allow_pickle=False) as b:
            mask = np.asarray(b["valid_water_mask"], dtype=bool)
            if bundle_key == "total_predictive_sd_chl":
                aleatory = b["aleatory_proxy_sd_chl"]
                epistemic = b["epistemic_proxy_sd_chl"]
                data_arr = np.sqrt(np.square(np.where(np.isnan(aleatory), 0, aleatory)) + np.square(np.where(np.isnan(epistemic), 0, epistemic)))
                data_arr = np.where(mask, data_arr, np.nan)
            else:
                data_arr = b[bundle_key]
                
            image = plot_map_layer(axis, data_arr, mask, base_water, lon, lat, cmap, vmin, vmax, iso_date(scene.date_token), BasemapManager(False, 0), False)
    for axis in flat_axes[len(scenes):]: axis.set_visible(False)
    if image is not None:
        cbar = fig.colorbar(image, ax=list(flat_axes[:len(scenes)]), shrink=0.72, pad=0.015, extend="neither")
        cbar.set_label(title, fontsize=11, fontweight="semibold")
    
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def draw_lognormal_ridge(ax: plt.Axes, x_pos: float, mu_s: float, sigma_t: float, color: str, width: float = 0.4):
    if not (np.isfinite(mu_s) and np.isfinite(sigma_t) and mu_s > 0):
        return
        
    sigma2_ln = np.log(1.0 + (sigma_t**2) / (mu_s**2))
    sigma_ln = np.sqrt(sigma2_ln)
    mu_ln = np.log(mu_s) - 0.5 * sigma2_ln
    
    y = np.linspace(0.01, mu_s + 4*sigma_t, 300)
    pdf = (1.0 / (y * sigma_ln * np.sqrt(2 * np.pi))) * np.exp(-((np.log(y) - mu_ln)**2) / (2 * sigma2_ln))
    
    pdf_max = np.max(pdf)
    if pdf_max > 0:
        pdf_norm = (pdf / pdf_max) * width
    else:
        pdf_norm = np.zeros_like(pdf)
        
    ax.fill_betweenx(y, x_pos - pdf_norm, x_pos + pdf_norm, color=color, alpha=0.5, edgecolor=color, linewidth=1.2)
    median_val = np.exp(mu_ln)
    ax.plot([x_pos - width*0.3, x_pos + width*0.3], [median_val, median_val], color='black', linewidth=1.5, zorder=5)

def draw_domain_lognormal(ax: plt.Axes, scenes: Sequence[Scene], region_masks: dict) -> None:
    domain_mask = region_masks["Entire domain"]
    date_labels = []
    
    for i, scene in enumerate(scenes):
        date_str = iso_date(scene.date_token)
        date_labels.append(date_str)
        with np.load(scene.npz_path, allow_pickle=False) as b:
            valid_mask = b["valid_water_mask"] & domain_mask
            mean_vals = b["predictive_mean_chl"][valid_mask]
            
            aleatory = b["aleatory_proxy_sd_chl"][valid_mask]
            epistemic = b["epistemic_proxy_sd_chl"][valid_mask]
            total_sd_arr = np.sqrt(np.square(aleatory) + np.square(epistemic))
            
            mean_vals = mean_vals[np.isfinite(mean_vals)]
            total_sd_arr = total_sd_arr[np.isfinite(total_sd_arr)]
            
            if len(mean_vals) > 0:
                mu_s = float(np.mean(mean_vals))
                sigma_t = float(np.sqrt(np.var(mean_vals) + np.mean(total_sd_arr**2)))
                draw_lognormal_ridge(ax, i, mu_s, sigma_t, COLORS["Domain Mean"])

    ax.set_xlim(-0.6, len(scenes) - 0.4)
    ax.set_xticks(np.arange(len(scenes)))
    ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Concentration Magnitude (mg/m³)", fontsize=11, fontweight="semibold")
    ax.set_title("Entire Domain Theoretical LN Density Distributions", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    
    legend_elements = [
        Patch(facecolor=COLORS["Domain Mean"], alpha=0.5, edgecolor=COLORS["Domain Mean"], label="Probabilistic LN PDF"),
        plt.Line2D([0], [0], color='black', linewidth=1.5, label="Distribution Median")
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

def draw_regional_lognormal(ax: plt.Axes, scenes: Sequence[Scene], region_masks: dict) -> None:
    regions = ["Jadro Estuary", "Kastela Bay (Rest of Bay)", "Split-Brac Canal"]
    offsets = [-0.28, 0.0, 0.28]
    date_labels = [iso_date(s.date_token) for s in scenes]
    
    for i, scene in enumerate(scenes):
        with np.load(scene.npz_path, allow_pickle=False) as b:
            mask = b["valid_water_mask"]
            mean_chl = b["predictive_mean_chl"]
            aleatory = b["aleatory_proxy_sd_chl"]
            epistemic = b["epistemic_proxy_sd_chl"]
            total_sd_arr = np.sqrt(np.square(aleatory) + np.square(epistemic))
            
            for region, offset in zip(regions, offsets):
                valid = mask & region_masks[region]
                vals = mean_chl[valid]
                sd_vals = total_sd_arr[valid]
                
                vals = vals[np.isfinite(vals)]
                sd_vals = sd_vals[np.isfinite(sd_vals)]
                
                if len(vals) > 0:
                    mu_s = float(np.mean(vals))
                    sigma_t = float(np.sqrt(np.var(vals) + np.mean(sd_vals**2)))
                    draw_lognormal_ridge(ax, i + offset, mu_s, sigma_t, COLORS[region], width=0.14)

    ax.set_xlim(-0.6, len(scenes) - 0.4)
    ax.set_xticks(np.arange(len(scenes)))
    ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Concentration Magnitude (mg/m³)", fontsize=11, fontweight="semibold")
    ax.set_xlabel("Satellite Acquisition Date", fontsize=11, fontweight="semibold")
    ax.set_title("Spatial Sub-Regional LN Density Distributions", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    
    legend_elements = [Patch(facecolor=COLORS[r], alpha=0.5, edgecolor=COLORS[r], label=r) for r in regions]
    ax.legend(handles=legend_elements, title="Marine Management Regime", loc="upper right", framealpha=0.9)

def create_lognormal_distributions(scenes: Sequence[Scene], region_masks: dict, output_dir: Path, dpi: int):
    print("Generating Probabilistic Lognormal Density Plots...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 15), constrained_layout=True)
    draw_domain_lognormal(axes[0], scenes, region_masks); draw_regional_lognormal(axes[1], scenes, region_masks)
    axes[0].set_xticklabels([])
    axes[0].set_xlabel("")
    fig.suptitle("Probabilistic Chl-a Lognormal Densities", fontsize=17, fontweight="bold")
    fig.savefig(output_dir / "Chl_LognormalPDF_Combined_Panel.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
    draw_domain_lognormal(ax, scenes, region_masks); ax.set_xlabel("Satellite Acquisition Date", fontsize=11, fontweight="semibold")
    fig.savefig(output_dir / "Chl_LognormalPDF_Domain.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
    draw_regional_lognormal(ax, scenes, region_masks)
    fig.savefig(output_dir / "Chl_LognormalPDF_Regional.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def draw_domain_timeseries(ax: plt.Axes, scenes: Sequence[Scene], region_masks: dict) -> None:
    domain_mask = region_masks["Entire domain"]
    dates = []
    c_med, c_low, c_high = [], [], []
    s_med, s_low, s_high = [], [], []

    for scene in scenes:
        dates.append(iso_date(scene.date_token))
        with np.load(scene.npz_path, allow_pickle=False) as b:
            valid = b["valid_water_mask"] & domain_mask
            chl = b["predictive_mean_chl"][valid]
            al = b["aleatory_proxy_sd_chl"][valid]
            ep = b["epistemic_proxy_sd_chl"][valid]
            tot = np.sqrt(np.square(np.where(np.isnan(al), 0, al)) + np.square(np.where(np.isnan(ep), 0, ep)))

            c_valid = chl[np.isfinite(chl)]
            if c_valid.size > 0:
                p2, p50, p98 = np.percentile(c_valid, [2.5, 50.0, 97.5])
                c_low.append(p2); c_med.append(p50); c_high.append(p98)
            else:
                c_low.append(np.nan); c_med.append(np.nan); c_high.append(np.nan)

            t_valid = tot[np.isfinite(tot)]
            if t_valid.size > 0:
                p2, p50, p98 = np.percentile(t_valid, [2.5, 50.0, 97.5])
                s_low.append(p2); s_med.append(p50); s_high.append(p98)
            else:
                s_low.append(np.nan); s_med.append(np.nan); s_high.append(np.nan)

    x = np.arange(len(dates))
    ax.fill_between(x, c_low, c_high, color=COLORS["Domain Mean"], alpha=0.2, linewidth=0)
    ax.fill_between(x, s_low, s_high, color=COLORS["Domain SD"], alpha=0.2, linewidth=0)
    ax.plot(x, c_med, color=COLORS["Domain Mean"], marker='o', linewidth=2.5, label="Median Mean Chl-$\\alpha$")
    ax.plot(x, s_med, color=COLORS["Domain SD"], marker='s', linestyle='--', linewidth=2.5, label="Median Total SD")

    ax.set_ylim(0.0, 1.6)
    ax.set_xlim(-0.5, len(dates) - 0.5)
    ax.set_xticks(x); ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Concentration Magnitude (mg/m³)", fontsize=11, fontweight="semibold")
    ax.set_title("Entire Domain 95% Spatial Envelopes", fontsize=13, fontweight="bold")
    ax.grid(axis="both", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

def draw_regional_timeseries(ax: plt.Axes, scenes: Sequence[Scene], region_masks: dict) -> None:
    regions = ["Jadro Estuary", "Kastela Bay (Rest of Bay)", "Split-Brac Canal"]
    dates = [iso_date(s.date_token) for s in scenes]
    x = np.arange(len(dates))

    for region in regions:
        c_med, c_low, c_high = [], [], []
        for scene in scenes:
            with np.load(scene.npz_path, allow_pickle=False) as b:
                valid = b["valid_water_mask"] & region_masks[region]
                chl = b["predictive_mean_chl"][valid]
                c_valid = chl[np.isfinite(chl)]
                if c_valid.size > 0:
                    p2, p50, p98 = np.percentile(c_valid, [2.5, 50.0, 97.5])
                    c_low.append(p2); c_med.append(p50); c_high.append(p98)
                else:
                    c_low.append(np.nan); c_med.append(np.nan); c_high.append(np.nan)

        ax.fill_between(x, c_low, c_high, color=COLORS[region], alpha=0.15, linewidth=0)
        ax.plot(x, c_med, color=COLORS[region], marker='o', markersize=4, linewidth=2, label=region)

    ax.set_ylim(0.0, 1.6)
    ax.set_xlim(-0.5, len(dates) - 0.5)
    ax.set_xticks(x); ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Concentration Magnitude (mg/m³)", fontsize=11, fontweight="semibold")
    ax.set_xlabel("Satellite Acquisition Date", fontsize=11, fontweight="semibold")
    ax.set_title("Spatial Sub-Regional 95% Envelopes", fontsize=13, fontweight="bold")
    ax.grid(axis="both", linestyle=":", alpha=0.4)
    ax.legend(title="Marine Management Regime", loc="upper right", framealpha=0.9, fontsize=9)

def create_timeseries_plots(scenes: Sequence[Scene], region_masks: dict, output_dir: Path, dpi: int):
    print("Generating Time Series Line Plots...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 15), constrained_layout=True)
    draw_domain_timeseries(axes[0], scenes, region_masks); draw_regional_timeseries(axes[1], scenes, region_masks)
    axes[0].set_xticklabels([])
    axes[0].set_xlabel("")
    fig.suptitle("Chronological Trend of Spatial Performance Indexes", fontsize=17, fontweight="bold")
    fig.savefig(output_dir / "Chl_TimeSeries_Combined_Panel.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
    draw_domain_timeseries(ax, scenes, region_masks); ax.set_xlabel("Satellite Acquisition Date", fontsize=11, fontweight="semibold")
    fig.savefig(output_dir / "Chl_TimeSeries_Domain.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
    draw_regional_timeseries(ax, scenes, region_masks)
    fig.savefig(output_dir / "Chl_TimeSeries_Regional.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def create_uncertainty_barchart(scenes: Sequence[Scene], region_masks: dict, output_dir: Path, dpi: int):
    print("Generating Uncertainty Partition Bar Chart...")
    dates = []
    al_mean, ep_mean, tot_mean = [], [], []

    for scene in scenes:
        dates.append(iso_date(scene.date_token))
        with np.load(scene.npz_path, allow_pickle=False) as b:
            valid = b["valid_water_mask"] & region_masks["Entire domain"]
            al = b["aleatory_proxy_sd_chl"][valid]
            ep = b["epistemic_proxy_sd_chl"][valid]
            
            al_valid = al[np.isfinite(al)]
            ep_valid = ep[np.isfinite(ep)]
            
            m_al = np.mean(al_valid) if al_valid.size > 0 else 0
            m_ep = np.mean(ep_valid) if ep_valid.size > 0 else 0
            m_tot = np.sqrt(m_al**2 + m_ep**2)

            al_mean.append(m_al)
            ep_mean.append(m_ep)
            tot_mean.append(m_tot)

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    x = np.arange(len(dates))
    width = 0.25

    ax.bar(x - width, al_mean, width, label=r'Aleatory SD ($\sigma_{spec}$)', color=COLORS["Aleatory"], edgecolor='black', alpha=0.8)
    ax.bar(x, ep_mean, width, label='Epistemic SD', color=COLORS["Epistemic"], edgecolor='black', alpha=0.8)
    ax.bar(x + width, tot_mean, width, label=r'Total SD ($\sigma_{Total}$)', color=COLORS["TotalSD"], edgecolor='black', alpha=0.8)

    ax.set_ylim(0.0, 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Average Uncertainty [mg/m³]", fontsize=11, fontweight="semibold")
    ax.set_xlabel("Satellite Acquisition Date", fontsize=11, fontweight="semibold")
    ax.set_title("Uncertainty Partitioning: Aleatory vs Epistemic (Grouped Comparison)", fontsize=15, fontweight="bold")
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)

    fig.savefig(output_dir / "Uncertainty_Partition_BarChart.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def run(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).expanduser().resolve()
    inference_dir = project_dir / DEFAULT_INFERENCE_DIRNAME
    xlarge_dir = project_dir / DEFAULT_XLARGE_DIRNAME
    output_dir = project_dir / DEFAULT_OUTPUT_DIRNAME
    
    jadro_pol_path = project_dir / "JadroEstuaryPolygon.csv"
    kastela_pol_path = project_dir / "KastelaBayPolygonLarge.csv"
    
    for subdir in ["Atlases", "Lognormal_Density_Plots", "Scene_Panels", "Time_Series_Plots", "Bar_Charts"]: 
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
    scenes = load_manifest(inference_dir, inference_dir / DEFAULT_MANIFEST)
    print(f"Loaded {len(scenes)} scenes from {inference_dir.name}.")
    
    with np.load(scenes[0].npz_path, allow_pickle=False) as bundle:
        lon, lat = np.asarray(bundle["longitude"]), np.asarray(bundle["latitude"])
    base_water = np.load(xlarge_dir / "xl_water_mask.npy", allow_pickle=False).astype(bool)

    region_masks = create_region_masks(lon, lat, base_water, read_polygon(jadro_pol_path), read_polygon(kastela_pol_path))
    basemap = BasemapManager(enabled=not args.skip_basemap, zoom=args.basemap_zoom)

    print("Generating 4-panel maps for each overpass...")
    for scene in scenes:
        with np.load(scene.npz_path, allow_pickle=False) as bundle:
            create_scene_4panel(scene, output_dir / "Scene_Panels" / f"Panel_4x4_{scene.date_token}.png", bundle, lon, lat, base_water, args.chl_vmax, basemap, args.dpi)

    create_lognormal_distributions(scenes, region_masks, output_dir / "Lognormal_Density_Plots", args.dpi)
    create_timeseries_plots(scenes, region_masks, output_dir / "Time_Series_Plots", args.dpi)
    create_uncertainty_barchart(scenes, region_masks, output_dir / "Bar_Charts", args.dpi)

    print("Generating 22-scene Atlases...")
    create_atlas(scenes, "predictive_mean_chl", lon, lat, base_water, 0.0, 2.0, "turbo", "Predictive Mean Chl-a (mg/m³)", output_dir / "Atlases" / "Mean_Chl_Atlas.png", args.dpi)
    create_atlas(scenes, "total_predictive_sd_chl", lon, lat, base_water, 0.0, 0.4, "turbo", "Total Predictive SD (mg/m³)", output_dir / "Atlases" / "Total_SD_Atlas.png", args.dpi)
    create_atlas(scenes, "spectral_trust_index", lon, lat, base_water, 0.0, 1.0, "viridis", "Spectral Trust Index (A_spec)", output_dir / "Atlases" / "Trust_Index_Atlas.png", args.dpi)

    print(f"\nSUCCESS: All publication figures exported to:\n  {output_dir}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--chl-vmax", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--basemap-zoom", type=int, default=11)
    parser.add_argument("--skip-basemap", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    run(parse_args())