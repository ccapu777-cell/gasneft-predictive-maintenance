"""
Gradient Boosting Models
=========================
XGBoost for:
  1. Binary classification — will failure occur within N cycles?
  2. RUL regression — how many cycles until failure?

Includes feature importance extraction and SHAP-ready outputs.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
import joblib
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Train XGBoost binary classifier for failure prediction.

    Returns (model, metrics_dict)
    """
    params = params or config.XGBOOST_PARAMS.copy()

    model = xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── Evaluate ───────────────────────────────────────
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "auc_roc": float(roc_auc_score(y_val, y_proba)),
        "accuracy": float((y_pred == y_val).mean()),
        "precision": float(classification_report(y_val, y_pred, output_dict=True)["1"]["precision"]),
        "recall": float(classification_report(y_val, y_pred, output_dict=True)["1"]["recall"]),
        "f1": float(classification_report(y_val, y_pred, output_dict=True)["1"]["f1-score"]),
    }

    print(f"[XGB-CLF] AUC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f}")
    print(classification_report(y_val, y_pred, target_names=["Healthy", "Failure"]))

    return model, metrics


def train_xgboost_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
) -> tuple[xgb.XGBRegressor, dict]:
    """
    Train XGBoost regressor for RUL prediction.

    Returns (model, metrics_dict)
    """
    params = params or config.XGBOOST_PARAMS.copy()

    model = xgb.XGBRegressor(
        **params,
        objective="reg:squarederror",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)

    metrics = {
        "mae": float(mean_absolute_error(y_val, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
        "r2": float(r2_score(y_val, y_pred)),
    }

    print(f"[XGB-REG] MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | R²: {metrics['r2']:.4f}")

    return model, metrics


def get_feature_importance(model, feature_names: list[str], top_n: int = 20) -> pd.DataFrame:
    """Extract and rank feature importances."""
    importance = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).head(top_n)
    return fi


def save_model(model, name: str):
    """Save model to registry."""
    out = Path(config.MODELS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"[SAVE] {path}")
    return path
