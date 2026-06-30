"""
Anomaly Detection Module
==========================
Two complementary approaches:
  1. Isolation Forest — fast, interpretable, no training labels needed
  2. Autoencoder — learns normal patterns, flags reconstruction errors

Used for real-time health monitoring: detect when equipment behavior
deviates from learned "healthy" baselines.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


# ═══════════════════════════════════════════════════════
#  Isolation Forest
# ═══════════════════════════════════════════════════════

def train_anomaly_detector(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    params: dict = None,
) -> tuple[IsolationForest, dict]:
    """
    Train Isolation Forest for unsupervised anomaly detection.
    Validates against failure labels if provided.
    """
    params = params or config.ANOMALY_PARAMS.copy()

    model = IsolationForest(
        contamination=params["contamination"],
        n_estimators=params["n_estimators"],
        random_state=params["random_state"],
        n_jobs=-1,
    )

    model.fit(X_train)

    metrics = {}
    if X_val is not None and y_val is not None:
        scores = model.decision_function(X_val)
        preds = model.predict(X_val)
        # IsolationForest: -1 = anomaly, 1 = normal → convert to 0/1
        anomaly_labels = (preds == -1).astype(int)

        metrics = {
            "anomaly_rate": float(anomaly_labels.mean()),
            "precision": float(precision_score(y_val, anomaly_labels, zero_division=0)),
            "recall": float(recall_score(y_val, anomaly_labels, zero_division=0)),
            "f1": float(f1_score(y_val, anomaly_labels, zero_division=0)),
            "mean_anomaly_score": float(scores[anomaly_labels == 1].mean()) if anomaly_labels.sum() > 0 else 0.0,
        }
        print(f"[IFOREST] Anomaly rate: {metrics['anomaly_rate']:.3f} | "
              f"F1 vs failures: {metrics['f1']:.4f}")

    return model, metrics


# ═══════════════════════════════════════════════════════
#  Autoencoder
# ═══════════════════════════════════════════════════════

class SensorAutoencoder(nn.Module):
    """
    Symmetric autoencoder for sensor reconstruction.
    High reconstruction error = anomalous behavior.
    """

    def __init__(self, input_dim: int, encoding_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = ((x - reconstructed) ** 2).mean(dim=1)
        return error


def train_autoencoder(
    X_train_healthy: pd.DataFrame,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> tuple[SensorAutoencoder, dict, float]:
    """
    Train autoencoder on HEALTHY data only.
    Learns normal operating patterns; anomalies have high reconstruction error.

    Returns (model, metrics, threshold)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_healthy.shape[1]

    train_tensor = torch.tensor(X_train_healthy.values, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor, train_tensor),
        batch_size=batch_size, shuffle=True,
    )

    model = SensorAutoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        losses = []
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            reconstructed = model(X_batch)
            loss = criterion(reconstructed, X_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"  AE Epoch {epoch+1}/{epochs} | Loss: {np.mean(losses):.6f}")

    # ── Compute anomaly threshold ──────────────────────
    model.eval()
    train_errors = model.get_reconstruction_error(train_tensor.to(device)).cpu().numpy()
    threshold = float(np.percentile(train_errors, 95))  # 95th percentile as baseline

    metrics = {"train_recon_error_mean": float(train_errors.mean()), "threshold": threshold}

    if X_val is not None and y_val is not None:
        val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
        val_errors = model.get_reconstruction_error(val_tensor).cpu().numpy()
        anomaly_preds = (val_errors > threshold).astype(int)

        metrics.update({
            "val_anomaly_rate": float(anomaly_preds.mean()),
            "val_precision": float(precision_score(y_val, anomaly_preds, zero_division=0)),
            "val_recall": float(recall_score(y_val, anomaly_preds, zero_division=0)),
            "val_f1": float(f1_score(y_val, anomaly_preds, zero_division=0)),
        })
        print(f"[AUTOENC] Threshold: {threshold:.4f} | F1: {metrics['val_f1']:.4f}")

    return model, metrics, threshold


def save_anomaly_model(model, name: str):
    """Save anomaly detection model."""
    out = Path(config.MODELS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(model, IsolationForest):
        path = out / f"{name}.joblib"
        joblib.dump(model, path)
    else:
        path = out / f"{name}.pt"
        torch.save(model.state_dict(), path)
    print(f"[SAVE] {path}")
    return path
