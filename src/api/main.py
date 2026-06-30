"""
FastAPI Inference API
======================
Production-ready REST endpoint for real-time predictions.

Endpoints:
  POST /predict/failure    — Binary failure prediction (within N cycles)
  POST /predict/rul        — Remaining Useful Life regression
  POST /predict/anomaly    — Anomaly score
  GET  /health             — Health check + model versions
  GET  /models             — List loaded models
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.data.preprocessing import build_features
from src.models.lstm_model import LSTMPredictor

# ═══════════════════════════════════════════════════════
#  App Setup
# ═══════════════════════════════════════════════════════

app = FastAPI(
    title="Gasneft Predictive Maintenance API",
    description="Real-time equipment health prediction for gas infrastructure",
    version="1.0.0",
)

# ── Load models at startup ─────────────────────────────
models = {}
model_dir = Path(config.MODELS_DIR)


@app.on_event("startup")
def load_models():
    """Load all trained models into memory."""
    try:
        models["scaler"] = joblib.load(model_dir / "scaler.joblib")
        models["xgb_classifier"] = joblib.load(model_dir / "xgb_classifier.joblib")
        models["xgb_regressor"] = joblib.load(model_dir / "xgb_regressor.joblib")
        models["isolation_forest"] = joblib.load(model_dir / "isolation_forest.joblib")
        print(f"[API] Loaded {len(models)} models from {model_dir}")
    except FileNotFoundError as e:
        print(f"[API] Warning: Some models not found — {e}")
        print("[API] Run training first: python -m src.models.trainer")


# ═══════════════════════════════════════════════════════
#  Request / Response Schemas
# ═══════════════════════════════════════════════════════

class SensorReading(BaseModel):
    """Single timestep sensor reading from equipment."""
    unit_id: int = Field(..., description="Equipment unit identifier")
    cycle: int = Field(..., description="Current operational cycle")
    setting_1: float = Field(0.0, description="Operational setting 1 (altitude)")
    setting_2: float = Field(0.0, description="Operational setting 2 (Mach)")
    setting_3: float = Field(80.0, description="Operational setting 3 (TRA)")
    sensor_1: float = Field(518.67)
    sensor_2: float = Field(642.7)
    sensor_3: float = Field(1590.0)
    sensor_4: float = Field(1408.0)
    sensor_5: float = Field(14.62)
    sensor_6: float = Field(21.61)
    sensor_7: float = Field(554.0)
    sensor_8: float = Field(2388.0)
    sensor_9: float = Field(9046.0)
    sensor_10: float = Field(1.3)
    sensor_11: float = Field(47.5)
    sensor_12: float = Field(522.0)
    sensor_13: float = Field(2388.0)
    sensor_14: float = Field(8140.0)
    sensor_15: float = Field(8.44)
    sensor_16: float = Field(0.03)
    sensor_17: float = Field(393.0)
    sensor_18: float = Field(2388.0)
    sensor_19: float = Field(100.0)
    sensor_20: float = Field(39.1)
    sensor_21: float = Field(23.42)

    class Config:
        json_schema_extra = {
            "example": {
                "unit_id": 1, "cycle": 150,
                "setting_1": 20.0, "setting_2": 0.7, "setting_3": 80.0,
                "sensor_1": 518.67, "sensor_2": 645.2, "sensor_3": 1598.0,
                "sensor_4": 1412.0, "sensor_5": 14.62, "sensor_6": 21.8,
                "sensor_7": 557.0, "sensor_8": 2388.0, "sensor_9": 9060.0,
                "sensor_10": 1.3, "sensor_11": 47.9, "sensor_12": 525.0,
                "sensor_13": 2388.0, "sensor_14": 8155.0, "sensor_15": 8.5,
                "sensor_16": 0.03, "sensor_17": 395.0, "sensor_18": 2388.0,
                "sensor_19": 100.0, "sensor_20": 39.3, "sensor_21": 23.6,
            }
        }


class FailurePrediction(BaseModel):
    unit_id: int
    failure_probability: float
    will_fail_soon: bool
    threshold_cycles: int
    confidence: str
    timestamp: str


class RULPrediction(BaseModel):
    unit_id: int
    predicted_rul: float
    risk_level: str
    timestamp: str


class AnomalyResult(BaseModel):
    unit_id: int
    anomaly_score: float
    is_anomalous: bool
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    version: str


# ═══════════════════════════════════════════════════════
#  Helper
# ═══════════════════════════════════════════════════════

def _reading_to_df(reading: SensorReading) -> pd.DataFrame:
    """Convert API request to DataFrame row."""
    data = reading.model_dump()
    return pd.DataFrame([data])


def _preprocess_single(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal preprocessing for single-row inference."""
    scaler = models.get("scaler")
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")

    # Add minimal features (no rolling for single row)
    for col in config.SENSOR_COLS:
        df[f"{col}_ewma"] = df[col]  # degenerate for single point
        for w in config.ROLLING_WINDOWS:
            df[f"{col}_rmean_{w}"] = df[col]
            df[f"{col}_rstd_{w}"] = 0.0

    # Sensor ratios
    eps = 1e-8
    df["ratio_T30_T50"] = df["sensor_3"] / (df["sensor_4"] + eps)
    df["ratio_P30_P15"] = df["sensor_7"] / (df["sensor_6"] + eps)
    df["ratio_Nc_Nf"] = df["sensor_9"] / (df["sensor_8"] + eps)
    df["ratio_BPR_phi"] = df["sensor_15"] / (df["sensor_12"] + eps)
    df["cycle_norm"] = 0.5  # unknown in single-row context

    feature_cols = [c for c in df.columns if c not in ("unit_id", "cycle", "RUL", "label", "cycle_norm")]
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


# ═══════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="operational",
        models_loaded=list(models.keys()),
        version="1.0.0",
    )


@app.get("/models")
def list_models():
    return {"models": list(models.keys()), "model_dir": str(model_dir)}


@app.post("/predict/failure", response_model=FailurePrediction)
def predict_failure(reading: SensorReading):
    """Predict if equipment will fail within threshold cycles."""
    if "xgb_classifier" not in models:
        raise HTTPException(status_code=503, detail="Classifier not loaded. Run training first.")

    df = _reading_to_df(reading)
    df = _preprocess_single(df)
    feature_cols = [c for c in df.columns if c not in ("unit_id", "cycle", "RUL", "label", "cycle_norm")]

    proba = float(models["xgb_classifier"].predict_proba(df[feature_cols])[0, 1])

    return FailurePrediction(
        unit_id=reading.unit_id,
        failure_probability=round(proba, 4),
        will_fail_soon=proba > 0.5,
        threshold_cycles=config.FAILURE_THRESHOLD,
        confidence="high" if abs(proba - 0.5) > 0.3 else "medium" if abs(proba - 0.5) > 0.1 else "low",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict/rul", response_model=RULPrediction)
def predict_rul(reading: SensorReading):
    """Predict Remaining Useful Life in cycles."""
    if "xgb_regressor" not in models:
        raise HTTPException(status_code=503, detail="Regressor not loaded. Run training first.")

    df = _reading_to_df(reading)
    df = _preprocess_single(df)
    feature_cols = [c for c in df.columns if c not in ("unit_id", "cycle", "RUL", "label", "cycle_norm")]

    rul = float(models["xgb_regressor"].predict(df[feature_cols])[0])
    rul = max(0, rul)

    if rul < 30:
        risk = "CRITICAL"
    elif rul < 60:
        risk = "HIGH"
    elif rul < 100:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return RULPrediction(
        unit_id=reading.unit_id,
        predicted_rul=round(rul, 1),
        risk_level=risk,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict/anomaly", response_model=AnomalyResult)
def predict_anomaly(reading: SensorReading):
    """Detect anomalous sensor behavior."""
    if "isolation_forest" not in models:
        raise HTTPException(status_code=503, detail="Anomaly detector not loaded. Run training first.")

    df = _reading_to_df(reading)
    df = _preprocess_single(df)
    feature_cols = [c for c in df.columns if c not in ("unit_id", "cycle", "RUL", "label", "cycle_norm")]

    score = float(models["isolation_forest"].decision_function(df[feature_cols])[0])
    prediction = int(models["isolation_forest"].predict(df[feature_cols])[0])

    return AnomalyResult(
        unit_id=reading.unit_id,
        anomaly_score=round(score, 4),
        is_anomalous=prediction == -1,
        timestamp=datetime.utcnow().isoformat(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
