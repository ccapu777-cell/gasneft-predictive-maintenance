"""
Exploratory Data Analysis — Gasneft Predictive Maintenance
============================================================
Run: python notebooks/eda.py

Generates EDA visualizations and saves them to data/eda_plots/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.data.loader import load_train_test
from src.data.preprocessing import add_rul_column

# ── Setup ──────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
PLOT_DIR = Path(config.DATA_DIR) / "eda_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 50)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    # Load data
    train_df, test_df, rul_true = load_train_test()
    train_df = add_rul_column(train_df)

    print(f"\nDataset shape: {train_df.shape}")
    print(f"Units: {train_df['unit_id'].nunique()}")
    print(f"Cycles range: {train_df['cycle'].min()} - {train_df['cycle'].max()}")

    # ── 1. Unit Lifetime Distribution ──────────────────
    print("\n[1/6] Unit lifetime distribution...")
    lifetimes = train_df.groupby("unit_id")["cycle"].max()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lifetimes, bins=25, edgecolor='white', alpha=0.8, color='#667eea')
    ax.axvline(lifetimes.mean(), color='red', linestyle='--', label=f'Mean: {lifetimes.mean():.0f}')
    ax.axvline(lifetimes.median(), color='orange', linestyle='--', label=f'Median: {lifetimes.median():.0f}')
    ax.set_xlabel("Total Operational Cycles")
    ax.set_ylabel("Number of Units")
    ax.set_title("Engine Unit Lifetime Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_lifetime_distribution.png", dpi=150)
    plt.close()

    # ── 2. Sensor Distributions ────────────────────────
    print("[2/6] Sensor distributions...")
    fig, axes = plt.subplots(3, 7, figsize=(20, 10))
    for i, col in enumerate(config.SENSOR_COLS):
        ax = axes[i // 7][i % 7]
        ax.hist(train_df[col], bins=40, alpha=0.7, color='#764ba2', edgecolor='white')
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("Sensor Value Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_sensor_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── 3. Sensor Trends for Sample Units ──────────────
    print("[3/6] Sensor degradation trends...")
    sample_units = sorted(train_df["unit_id"].unique())[:4]
    key_sensors = ["sensor_2", "sensor_3", "sensor_7", "sensor_11", "sensor_15"]

    fig, axes = plt.subplots(len(key_sensors), 1, figsize=(14, 3 * len(key_sensors)), sharex=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(sample_units)))

    for j, sensor in enumerate(key_sensors):
        for k, uid in enumerate(sample_units):
            unit = train_df[train_df["unit_id"] == uid]
            axes[j].plot(unit["cycle"], unit[sensor], alpha=0.7, color=colors[k], label=f"Unit {uid}")
        axes[j].set_ylabel(sensor)
        if j == 0:
            axes[j].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Operational Cycle")
    fig.suptitle("Sensor Degradation Trends (Sample Units)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_degradation_trends.png", dpi=150)
    plt.close()

    # ── 4. Correlation Matrix ──────────────────────────
    print("[4/6] Sensor correlation matrix...")
    corr = train_df[config.SENSOR_COLS].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                annot=False, square=True, linewidths=0.5, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title("Sensor Correlation Matrix")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_correlation_matrix.png", dpi=150)
    plt.close()

    # ── 5. RUL Distribution ───────────────────────────
    print("[5/6] RUL distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(train_df["RUL"], bins=50, edgecolor='white', alpha=0.8, color='#e74c3c')
    axes[0].axvline(config.RUL_CLIP, color='black', linestyle='--', label=f'Clip at {config.RUL_CLIP}')
    axes[0].set_xlabel("RUL (cycles)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("RUL Distribution (Clipped)")
    axes[0].legend()

    # Percentage of data in failure zone
    fail_pct = (train_df["RUL"] <= config.FAILURE_THRESHOLD).mean() * 100
    axes[1].bar(["Healthy", "Near Failure"],
                [100 - fail_pct, fail_pct],
                color=["#00CC66", "#FF4B4B"], edgecolor='white')
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title(f"Class Balance (threshold = {config.FAILURE_THRESHOLD} cycles)")
    for i, v in enumerate([100 - fail_pct, fail_pct]):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_rul_distribution.png", dpi=150)
    plt.close()

    # ── 6. Sensor vs RUL Scatter ──────────────────────
    print("[6/6] Sensor vs RUL relationships...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    scatter_sensors = ["sensor_2", "sensor_3", "sensor_7", "sensor_11", "sensor_15", "sensor_21"]

    for i, sensor in enumerate(scatter_sensors):
        ax = axes[i // 3][i % 3]
        sample = train_df.sample(min(5000, len(train_df)), random_state=42)
        ax.scatter(sample["RUL"], sample[sensor], alpha=0.15, s=5, color='#667eea')
        ax.set_xlabel("RUL")
        ax.set_ylabel(sensor)
        ax.set_title(f"{sensor} vs RUL")

    fig.suptitle("Key Sensor Values vs Remaining Useful Life", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_sensor_vs_rul.png", dpi=150)
    plt.close()

    print(f"\n✅ All plots saved to {PLOT_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
