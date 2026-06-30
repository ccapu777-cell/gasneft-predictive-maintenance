"""
Data Loader
============
Load train/test splits from CSV or generate on the fly.
"""

import pandas as pd
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.data.generator import generate_cmapss_dataset


def load_train_test(data_dir: str = None, regenerate: bool = False):
    """
    Load (or generate) train/test data.

    Returns
    -------
    train_df, test_df, rul_true
    """
    data_dir = Path(data_dir or config.DATA_DIR)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    rul_path = data_dir / "rul_test.csv"

    if not train_path.exists() or regenerate:
        print("[LOADER] Generating synthetic dataset...")
        return generate_cmapss_dataset(save=True)

    print(f"[LOADER] Loading from {data_dir}/")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    rul_true = pd.read_csv(rul_path).squeeze("columns")

    return train_df, test_df, rul_true


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature columns (exclude meta + targets)."""
    exclude = {"unit_id", "cycle", "RUL", "label", "cycle_norm"}
    return [c for c in df.columns if c not in exclude]
