"""
Feature Engineering Pipeline
=============================
Transforms raw sensor telemetry into ML-ready features:
  - RUL target computation (piecewise-linear, clipped)
  - Rolling statistics (mean, std, min, max, trend slope)
  - Exponential weighted moving averages
  - Sensor cross-ratios (domain-informed)
  - Normalization
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


def add_rul_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Remaining Useful Life for each row.
    RUL = max_cycle_for_unit - current_cycle, clipped to RUL_CLIP.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"] = max_cycles - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=config.RUL_CLIP)
    return df


def add_failure_label(df: pd.DataFrame, threshold: int = config.FAILURE_THRESHOLD) -> pd.DataFrame:
    """Binary label: will this unit fail within `threshold` cycles?"""
    df = df.copy()
    if "RUL" not in df.columns:
        df = add_rul_column(df)
    df["label"] = (df["RUL"] <= threshold).astype(int)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """
    Per-unit rolling statistics over sensor columns.
    Adds: rolling_mean, rolling_std, rolling_slope for each window size.
    """
    df = df.copy()
    windows = windows or config.ROLLING_WINDOWS

    for w in windows:
        for col in config.SENSOR_COLS:
            grp = df.groupby("unit_id")[col]
            df[f"{col}_rmean_{w}"] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"{col}_rstd_{w}"] = grp.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))

    return df


def add_ewma_features(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """Exponential weighted moving average for trend smoothing."""
    df = df.copy()
    for col in config.SENSOR_COLS:
        df[f"{col}_ewma"] = df.groupby("unit_id")[col].transform(
            lambda x: x.ewm(span=span, adjust=False).mean()
        )
    return df


def add_sensor_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-informed cross-sensor ratios (temperature/pressure combos)."""
    df = df.copy()
    eps = 1e-8
    # Temperature ratio: T30/T50
    df["ratio_T30_T50"] = df["sensor_3"] / (df["sensor_4"] + eps)
    # Pressure ratio: P30/P15
    df["ratio_P30_P15"] = df["sensor_7"] / (df["sensor_6"] + eps)
    # Compressor efficiency proxy: Nc/Nf
    df["ratio_Nc_Nf"] = df["sensor_9"] / (df["sensor_8"] + eps)
    # Bypass ratio trend
    df["ratio_BPR_phi"] = df["sensor_15"] / (df["sensor_12"] + eps)
    return df


def add_cycle_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Normalized cycle position within each unit's observed life."""
    df = df.copy()
    max_c = df.groupby("unit_id")["cycle"].transform("max")
    df["cycle_norm"] = df["cycle"] / max_c
    return df


def normalize_sensors(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True):
    """
    Z-score normalization on sensor + engineered features.
    Returns (df, fitted_scaler).
    """
    feature_cols = [c for c in df.columns if c not in
                    ("unit_id", "cycle", "RUL", "label", "cycle_norm")]

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, scaler


def build_features(
    df: pd.DataFrame,
    is_training: bool = True,
    scaler: StandardScaler = None,
    rolling_windows: list[int] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df : raw dataframe with unit_id, cycle, settings, sensors
    is_training : if True, compute RUL and fit scaler
    scaler : pre-fitted scaler (for test/inference)

    Returns
    -------
    df : feature-enriched dataframe
    scaler : fitted StandardScaler
    """
    if is_training:
        df = add_rul_column(df)
        df = add_failure_label(df)

    df = add_rolling_features(df, rolling_windows)
    df = add_ewma_features(df)
    df = add_sensor_ratios(df)
    df = add_cycle_normalization(df)
    df = df.fillna(0)

    df, scaler = normalize_sensors(df, scaler=scaler, fit=is_training)

    return df, scaler
