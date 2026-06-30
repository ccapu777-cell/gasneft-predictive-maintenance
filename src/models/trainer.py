"""
Training Orchestrator
======================
End-to-end pipeline: data → features → train all models → log to MLflow.

Usage:
    python -m src.models.trainer
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import mlflow
import mlflow.xgboost
import joblib
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.data.loader import load_train_test, get_feature_columns
from src.data.preprocessing import build_features
from src.models.gradient_boost import (
    train_xgboost_classifier, train_xgboost_regressor,
    get_feature_importance, save_model,
)
from src.models.lstm_model import train_lstm, save_lstm
from src.models.anomaly import train_anomaly_detector, train_autoencoder, save_anomaly_model


def split_by_unit(df: pd.DataFrame, test_size: float = 0.2, seed: int = config.RANDOM_SEED):
    """Split data by unit_id to prevent leakage."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df["unit_id"]
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def run_full_pipeline():
    """Execute complete training pipeline."""

    print("=" * 60)
    print("  GASNEFT PREDICTIVE MAINTENANCE — TRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Load Data ───────────────────────────────────
    print("\n[1/6] Loading data...")
    train_raw, test_raw, rul_true = load_train_test()
    print(f"  Train: {len(train_raw)} rows, {train_raw['unit_id'].nunique()} units")
    print(f"  Test:  {len(test_raw)} rows, {test_raw['unit_id'].nunique()} units")

    # ── 2. Feature Engineering ─────────────────────────
    print("\n[2/6] Building features...")
    train_feat, scaler = build_features(train_raw, is_training=True)
    joblib.dump(scaler, Path(config.MODELS_DIR) / "scaler.joblib")

    # Train/val split by unit
    train_df, val_df = split_by_unit(train_feat)
    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    y_train_rul = train_df["RUL"]
    y_val_rul = val_df["RUL"]
    y_train_label = train_df["label"]
    y_val_label = val_df["label"]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Train split: {len(train_df)} | Val split: {len(val_df)}")

    # ── Setup MLflow ───────────────────────────────────
    db_path = config.MLRUNS_DIR / "mlflow.db"
    config.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("gasneft-predictive-maintenance")

    # ── 3. XGBoost Classifier ──────────────────────────
    print("\n[3/6] Training XGBoost Classifier...")
    with mlflow.start_run(run_name="xgboost_classifier"):
        xgb_clf, clf_metrics = train_xgboost_classifier(
            X_train, y_train_label, X_val, y_val_label
        )
        mlflow.log_params(config.XGBOOST_PARAMS)
        mlflow.log_metrics(clf_metrics)
        mlflow.xgboost.log_model(xgb_clf, "xgboost_classifier")
        save_model(xgb_clf, "xgb_classifier")

        fi = get_feature_importance(xgb_clf, feature_cols)
        fi.to_csv(Path(config.MODELS_DIR) / "feature_importance_clf.csv", index=False)

    # ── 4. XGBoost Regressor ───────────────────────────
    print("\n[4/6] Training XGBoost Regressor (RUL)...")
    with mlflow.start_run(run_name="xgboost_regressor"):
        xgb_reg, reg_metrics = train_xgboost_regressor(
            X_train, y_train_rul, X_val, y_val_rul
        )
        mlflow.log_params(config.XGBOOST_PARAMS)
        mlflow.log_metrics(reg_metrics)
        mlflow.xgboost.log_model(xgb_reg, "xgboost_regressor")
        save_model(xgb_reg, "xgb_regressor")

        fi_reg = get_feature_importance(xgb_reg, feature_cols)
        fi_reg.to_csv(Path(config.MODELS_DIR) / "feature_importance_reg.csv", index=False)

    # ── 5. LSTM ────────────────────────────────────────
    print("\n[5/6] Training LSTM...")
    with mlflow.start_run(run_name="lstm_rul"):
        lstm_model, lstm_metrics = train_lstm(train_df, val_df, feature_cols)
        mlflow.log_params(config.LSTM_PARAMS)
        history = lstm_metrics.pop("history", {})
        mlflow.log_metrics(lstm_metrics)
        save_lstm(lstm_model, "lstm_rul")

    # ── 6. Anomaly Detection ──────────────────────────
    print("\n[6/6] Training Anomaly Detectors...")

    # 6a. Isolation Forest
    with mlflow.start_run(run_name="isolation_forest"):
        iforest, iforest_metrics = train_anomaly_detector(
            X_train, X_val, y_val_label
        )
        mlflow.log_params(config.ANOMALY_PARAMS)
        mlflow.log_metrics(iforest_metrics)
        save_anomaly_model(iforest, "isolation_forest")

    # 6b. Autoencoder (train on healthy samples only)
    healthy_mask = y_train_label == 0
    with mlflow.start_run(run_name="autoencoder"):
        ae_model, ae_metrics, ae_threshold = train_autoencoder(
            X_train[healthy_mask], X_val, y_val_label
        )
        mlflow.log_metrics(ae_metrics)
        save_anomaly_model(ae_model, "autoencoder")
        joblib.dump(ae_threshold, Path(config.MODELS_DIR) / "ae_threshold.joblib")

    # ── Summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  XGBoost Classifier:  AUC={clf_metrics['auc_roc']:.4f}  F1={clf_metrics['f1']:.4f}")
    print(f"  XGBoost Regressor:   MAE={reg_metrics['mae']:.2f}  RMSE={reg_metrics['rmse']:.2f}  R²={reg_metrics['r2']:.4f}")
    print(f"  LSTM RUL:            MAE={lstm_metrics['mae']:.2f}  RMSE={lstm_metrics['rmse']:.2f}")
    print(f"  Isolation Forest:    F1={iforest_metrics.get('f1', 'N/A')}")
    print(f"  Autoencoder:         F1={ae_metrics.get('val_f1', 'N/A')}")
    print(f"\n  Models saved to:     {config.MODELS_DIR}/")
    print(f"  MLflow experiments:  {config.MLRUNS_DIR}/")
    print(f"\n  → Run 'mlflow ui --backend-store-uri sqlite:///{db_path}' to view experiments")
    print("=" * 60)

    return {
        "classifier": clf_metrics,
        "regressor": reg_metrics,
        "lstm": lstm_metrics,
        "isolation_forest": iforest_metrics,
        "autoencoder": ae_metrics,
    }


if __name__ == "__main__":
    run_full_pipeline()
