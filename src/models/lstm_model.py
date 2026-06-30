"""
LSTM Model for RUL Prediction
===============================
Sequence-to-one LSTM that takes a window of sensor readings
and predicts Remaining Useful Life.

Uses PyTorch for flexibility and production readiness.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


# ═══════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    """Sliding-window dataset for LSTM training."""

    def __init__(self, df: pd.DataFrame, feature_cols: list[str],
                 target_col: str = "RUL", seq_len: int = None):
        self.seq_len = seq_len or config.LSTM_PARAMS["sequence_length"]
        self.sequences = []
        self.targets = []

        for uid in df["unit_id"].unique():
            unit_data = df[df["unit_id"] == uid].sort_values("cycle")
            values = unit_data[feature_cols].values
            rul = unit_data[target_col].values

            for i in range(len(values) - self.seq_len + 1):
                self.sequences.append(values[i : i + self.seq_len])
                self.targets.append(rul[i + self.seq_len - 1])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.targets[idx]),
        )


# ═══════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════

class LSTMPredictor(nn.Module):
    """
    Stacked LSTM → FC for RUL regression.

    Architecture
    ------------
    Input: (batch, seq_len, n_features)
    → LSTM layers with dropout
    → Last hidden state
    → FC → ReLU → FC → output (scalar RUL)
    """

    def __init__(self, n_features: int, hidden_size: int = None,
                 num_layers: int = None, dropout: float = None):
        super().__init__()

        hidden_size = hidden_size or config.LSTM_PARAMS["hidden_size"]
        num_layers = num_layers or config.LSTM_PARAMS["num_layers"]
        dropout = dropout or config.LSTM_PARAMS["dropout"]

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # take last timestep
        return self.fc(last_hidden).squeeze(-1)


# ═══════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════

def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict = None,
) -> tuple[LSTMPredictor, dict]:
    """
    Train LSTM model with early stopping.

    Returns (model, metrics_dict)
    """
    params = params or config.LSTM_PARAMS.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Datasets ───────────────────────────────────────
    train_ds = SequenceDataset(train_df, feature_cols, seq_len=params["sequence_length"])
    val_ds = SequenceDataset(val_df, feature_cols, seq_len=params["sequence_length"])

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"])

    # ── Model ──────────────────────────────────────────
    model = LSTMPredictor(
        n_features=len(feature_cols),
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    # ── Training loop ──────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(params["epochs"]):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_losses.append(criterion(pred, y_batch).item())
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{params['epochs']} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # ── Final metrics ──────────────────────────────────
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    metrics = {
        "mae": float(np.mean(np.abs(all_preds - all_targets))),
        "rmse": float(np.sqrt(np.mean((all_preds - all_targets) ** 2))),
        "val_loss": float(best_val_loss),
        "epochs_trained": len(history["train_loss"]),
    }
    metrics["history"] = history

    print(f"[LSTM] MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f}")

    return model, metrics


def save_lstm(model: LSTMPredictor, name: str = "lstm_rul"):
    """Save PyTorch model."""
    out = Path(config.MODELS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pt"
    torch.save(model.state_dict(), path)
    print(f"[SAVE] {path}")
    return path
