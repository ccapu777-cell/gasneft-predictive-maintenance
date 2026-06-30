"""
Synthetic Turbofan Degradation Data Generator
==============================================
Generates sensor telemetry mimicking NASA C-MAPSS format:
  - Multiple engine units, each running until failure
  - 3 operational settings + 21 sensor channels
  - Realistic degradation trends + noise

This lets the project run standalone without downloading NASA data.
To use real C-MAPSS data, see README.md instructions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


def _generate_single_unit(unit_id: int, max_cycle: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate degradation trajectory for one engine unit."""
    cycles = np.arange(1, max_cycle + 1)
    n = len(cycles)

    # ── Operational settings (3 modes) ─────────────────
    settings = np.column_stack([
        rng.choice([0.0, 10.0, 20.0, 25.0, 42.0], size=n),          # altitude proxy
        rng.choice([0.0, 0.25, 0.50, 0.70, 0.84], size=n),          # Mach proxy
        rng.uniform(60, 100, size=n),                                 # TRA proxy
    ])

    # ── Sensor baselines + degradation ─────────────────
    # Each sensor has: baseline, degradation_rate, noise_std
    sensor_profiles = [
        (518.67, 0.00, 0.5),    # s1:  T2 total temp (stable)
        (642.70, 0.02, 1.2),    # s2:  T24
        (1590.0, 0.08, 3.0),    # s3:  T30
        (1408.0, 0.05, 2.5),    # s4:  T50
        (14.62,  0.00, 0.02),   # s5:  P2 (stable)
        (21.61,  0.01, 0.1),    # s6:  P15
        (554.0,  0.03, 1.0),    # s7:  P30
        (2388.0, 0.00, 0.5),    # s8:  Nf (stable)
        (9046.0, 0.03, 5.0),    # s9:  Nc
        (1.30,   0.00, 0.001),  # s10: epr (stable)
        (47.50,  0.02, 0.2),    # s11: Ps30
        (522.0,  0.04, 1.5),    # s12: phi
        (2388.0, 0.00, 0.3),    # s13: NRf (stable)
        (8140.0, 0.03, 4.0),    # s14: NRc
        (8.44,   0.005, 0.05),  # s15: BPR
        (0.03,   0.0001, 0.001),# s16: farB
        (393.0,  0.02, 1.0),    # s17: htBleed
        (2388.0, 0.00, 0.3),    # s18: Nf_dmd (stable)
        (100.0,  0.00, 0.01),   # s19: PCNfR_dmd (stable)
        (39.10,  0.01, 0.15),   # s20: W31
        (23.42,  0.02, 0.12),   # s21: W32
    ]

    sensors = np.zeros((n, config.N_SENSORS))
    for j, (baseline, deg_rate, noise_std) in enumerate(sensor_profiles):
        # Degradation accelerates near end-of-life (exponential ramp)
        progress = cycles / max_cycle
        degradation = baseline * deg_rate * (np.exp(3.0 * progress) - 1) / (np.e**3 - 1)
        # Some sensors decrease instead of increase
        direction = rng.choice([-1, 1])
        noise = rng.normal(0, noise_std, size=n)
        sensors[:, j] = baseline + direction * degradation + noise

    df = pd.DataFrame()
    df["cycle"] = cycles
    for k, col in enumerate(config.SETTING_COLS):
        df[col] = settings[:, k]
    for k, col in enumerate(config.SENSOR_COLS):
        df[col] = sensors[:, k]
    df.insert(0, "unit_id", unit_id)

    return df


def generate_cmapss_dataset(
    n_train: int = config.N_UNITS_TRAIN,
    n_test: int = config.N_UNITS_TEST,
    seed: int = config.RANDOM_SEED,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Generate full train/test datasets in C-MAPSS format.

    Returns
    -------
    train_df : full run-to-failure trajectories
    test_df  : truncated trajectories (partial life)
    rul_true : true RUL for each test unit at truncation point
    """
    rng = np.random.default_rng(seed)
    total = n_train + n_test

    # Random lifetimes per unit
    lifetimes = rng.integers(config.MIN_CYCLES, config.MAX_CYCLES, size=total)

    # ── Training set: full run-to-failure ──────────────
    train_frames = []
    for i in range(n_train):
        df = _generate_single_unit(i + 1, lifetimes[i], rng)
        train_frames.append(df)
    train_df = pd.concat(train_frames, ignore_index=True)

    # ── Test set: truncated + true RUL ─────────────────
    test_frames = []
    rul_values = []
    for i in range(n_test):
        idx = n_train + i
        full_life = lifetimes[idx]
        # Truncate at random point (40-90% of life)
        cutoff = int(full_life * rng.uniform(0.4, 0.9))
        cutoff = max(cutoff, 30)  # at least 30 cycles
        df = _generate_single_unit(i + 1, full_life, rng).iloc[:cutoff]
        test_frames.append(df)
        rul_values.append(full_life - cutoff)

    test_df = pd.concat(test_frames, ignore_index=True)
    rul_true = pd.Series(rul_values, name="RUL")

    if save:
        out = Path(config.DATA_DIR)
        out.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out / "train.csv", index=False)
        test_df.to_csv(out / "test.csv", index=False)
        rul_true.to_csv(out / "rul_test.csv", index=False)
        print(f"[DATA] Saved {len(train_df)} train rows, {len(test_df)} test rows → {out}/")

    return train_df, test_df, rul_true


if __name__ == "__main__":
    generate_cmapss_dataset()
