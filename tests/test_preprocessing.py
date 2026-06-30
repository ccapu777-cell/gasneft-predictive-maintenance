"""
Tests for data generation and preprocessing pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.data.generator import generate_cmapss_dataset
from src.data.preprocessing import (
    add_rul_column, add_failure_label, add_rolling_features,
    add_sensor_ratios, build_features,
)


class TestDataGeneration:
    """Tests for synthetic data generator."""

    def setup_method(self):
        self.train_df, self.test_df, self.rul_true = generate_cmapss_dataset(
            n_train=5, n_test=2, seed=42, save=False,
        )

    def test_train_shape(self):
        assert len(self.train_df) > 0
        assert "unit_id" in self.train_df.columns
        assert "cycle" in self.train_df.columns

    def test_sensor_columns_present(self):
        for col in config.SENSOR_COLS:
            assert col in self.train_df.columns, f"Missing sensor: {col}"

    def test_setting_columns_present(self):
        for col in config.SETTING_COLS:
            assert col in self.train_df.columns, f"Missing setting: {col}"

    def test_correct_number_of_train_units(self):
        assert self.train_df["unit_id"].nunique() == 5

    def test_correct_number_of_test_units(self):
        assert self.test_df["unit_id"].nunique() == 2

    def test_rul_true_length(self):
        assert len(self.rul_true) == 2

    def test_rul_positive(self):
        assert (self.rul_true >= 0).all()

    def test_cycles_start_at_one(self):
        for uid in self.train_df["unit_id"].unique():
            unit = self.train_df[self.train_df["unit_id"] == uid]
            assert unit["cycle"].min() == 1

    def test_no_nan_in_sensors(self):
        assert not self.train_df[config.SENSOR_COLS].isna().any().any()

    def test_reproducibility(self):
        train2, _, _ = generate_cmapss_dataset(n_train=5, n_test=2, seed=42, save=False)
        pd.testing.assert_frame_equal(self.train_df, train2)


class TestPreprocessing:
    """Tests for feature engineering pipeline."""

    def setup_method(self):
        self.train_df, _, _ = generate_cmapss_dataset(
            n_train=5, n_test=2, seed=42, save=False,
        )

    def test_rul_column(self):
        df = add_rul_column(self.train_df)
        assert "RUL" in df.columns
        assert (df["RUL"] >= 0).all()

    def test_rul_clipping(self):
        df = add_rul_column(self.train_df)
        assert df["RUL"].max() <= config.RUL_CLIP

    def test_rul_last_cycle_zero(self):
        df = add_rul_column(self.train_df)
        for uid in df["unit_id"].unique():
            unit = df[df["unit_id"] == uid]
            last_rul = unit.iloc[-1]["RUL"]
            assert last_rul == 0, f"Unit {uid}: last RUL should be 0, got {last_rul}"

    def test_failure_label(self):
        df = add_failure_label(self.train_df)
        assert "label" in df.columns
        assert set(df["label"].unique()) <= {0, 1}

    def test_failure_label_correct(self):
        df = add_failure_label(self.train_df, threshold=30)
        df_with_rul = add_rul_column(self.train_df)
        failing = df_with_rul["RUL"] <= 30
        assert (df["label"] == failing.astype(int)).all()

    def test_rolling_features(self):
        df = add_rolling_features(self.train_df, windows=[5])
        for col in config.SENSOR_COLS[:3]:
            assert f"{col}_rmean_5" in df.columns
            assert f"{col}_rstd_5" in df.columns

    def test_sensor_ratios(self):
        df = add_sensor_ratios(self.train_df)
        assert "ratio_T30_T50" in df.columns
        assert "ratio_P30_P15" in df.columns
        assert "ratio_Nc_Nf" in df.columns

    def test_build_features_no_nan(self):
        df, scaler = build_features(self.train_df, is_training=True, rolling_windows=[5])
        assert not df.isna().any().any(), f"NaN found in columns: {df.columns[df.isna().any()].tolist()}"

    def test_build_features_returns_scaler(self):
        df, scaler = build_features(self.train_df, is_training=True, rolling_windows=[5])
        assert scaler is not None

    def test_scaler_reuse(self):
        df1, scaler = build_features(self.train_df, is_training=True, rolling_windows=[5])
        # Simulate test preprocessing
        train2, _, _ = generate_cmapss_dataset(n_train=5, n_test=2, seed=99, save=False)
        df2 = add_rul_column(train2)
        df2 = add_failure_label(df2)
        df2 = add_rolling_features(df2, [5])
        from src.data.preprocessing import add_ewma_features, add_sensor_ratios as asr, add_cycle_normalization, normalize_sensors
        df2 = add_ewma_features(df2)
        df2 = asr(df2)
        df2 = add_cycle_normalization(df2)
        df2 = df2.fillna(0)
        df2, _ = normalize_sensors(df2, scaler=scaler, fit=False)
        assert not df2.isna().any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
