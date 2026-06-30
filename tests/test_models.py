"""
Tests for model training and inference.
"""

import pytest
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.data.generator import generate_cmapss_dataset
from src.data.preprocessing import build_features
from src.data.loader import get_feature_columns
from src.models.gradient_boost import train_xgboost_classifier, train_xgboost_regressor


@pytest.fixture(scope="module")
def prepared_data():
    """Generate and preprocess a small dataset for testing."""
    train_df, _, _ = generate_cmapss_dataset(n_train=10, n_test=3, seed=42, save=False)
    train_feat, scaler = build_features(train_df, is_training=True, rolling_windows=[5])

    # Split
    units = train_feat["unit_id"].unique()
    train_units = units[:8]
    val_units = units[8:]

    train_split = train_feat[train_feat["unit_id"].isin(train_units)]
    val_split = train_feat[train_feat["unit_id"].isin(val_units)]

    feature_cols = get_feature_columns(train_split)

    return {
        "X_train": train_split[feature_cols],
        "X_val": val_split[feature_cols],
        "y_train_label": train_split["label"],
        "y_val_label": val_split["label"],
        "y_train_rul": train_split["RUL"],
        "y_val_rul": val_split["RUL"],
        "feature_cols": feature_cols,
        "train_split": train_split,
        "val_split": val_split,
    }


class TestXGBoostClassifier:
    """Tests for failure classification model."""

    def test_trains_without_error(self, prepared_data):
        model, metrics = train_xgboost_classifier(
            prepared_data["X_train"], prepared_data["y_train_label"],
            prepared_data["X_val"], prepared_data["y_val_label"],
            params={**config.XGBOOST_PARAMS, "n_estimators": 10},
        )
        assert model is not None

    def test_returns_metrics(self, prepared_data):
        _, metrics = train_xgboost_classifier(
            prepared_data["X_train"], prepared_data["y_train_label"],
            prepared_data["X_val"], prepared_data["y_val_label"],
            params={**config.XGBOOST_PARAMS, "n_estimators": 10},
        )
        assert "auc_roc" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["auc_roc"] <= 1

    def test_predictions_shape(self, prepared_data):
        model, _ = train_xgboost_classifier(
            prepared_data["X_train"], prepared_data["y_train_label"],
            prepared_data["X_val"], prepared_data["y_val_label"],
            params={**config.XGBOOST_PARAMS, "n_estimators": 10},
        )
        preds = model.predict(prepared_data["X_val"])
        assert len(preds) == len(prepared_data["X_val"])
        assert set(preds) <= {0, 1}


class TestXGBoostRegressor:
    """Tests for RUL regression model."""

    def test_trains_without_error(self, prepared_data):
        model, metrics = train_xgboost_regressor(
            prepared_data["X_train"], prepared_data["y_train_rul"],
            prepared_data["X_val"], prepared_data["y_val_rul"],
            params={**config.XGBOOST_PARAMS, "n_estimators": 10},
        )
        assert model is not None

    def test_returns_regression_metrics(self, prepared_data):
        _, metrics = train_xgboost_regressor(
            prepared_data["X_train"], prepared_data["y_train_rul"],
            prepared_data["X_val"], prepared_data["y_val_rul"],
            params={**config.XGBOOST_PARAMS, "n_estimators": 10},
        )
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_predictions_reasonable(self, prepared_data):
        model, _ = train_xgboost_regressor(
            prepared_data["X_train"], prepared_data["y_train_rul"],
            prepared_data["X_val"], prepared_data["y_val_rul"],
            params={**config.XGBOOST_PARAMS, "n_estimators": 10},
        )
        preds = model.predict(prepared_data["X_val"])
        # RUL predictions should be within a reasonable range
        assert preds.min() >= -50  # small negative ok for regression
        assert preds.max() <= 200


class TestAnomalyDetection:
    """Tests for anomaly detection models."""

    def test_isolation_forest(self, prepared_data):
        from src.models.anomaly import train_anomaly_detector
        model, metrics = train_anomaly_detector(
            prepared_data["X_train"],
            prepared_data["X_val"],
            prepared_data["y_val_label"],
        )
        assert model is not None
        assert "anomaly_rate" in metrics

    def test_isolation_forest_predictions(self, prepared_data):
        from src.models.anomaly import train_anomaly_detector
        model, _ = train_anomaly_detector(prepared_data["X_train"])
        preds = model.predict(prepared_data["X_val"])
        assert set(preds) <= {-1, 1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
