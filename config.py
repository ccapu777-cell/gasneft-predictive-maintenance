"""
Gasneft Predictive Maintenance — Central Configuration
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models_registry"
MLRUNS_DIR = ROOT_DIR / "mlruns"

# ── Data Generation ────────────────────────────────────
N_UNITS_TRAIN = 80
N_UNITS_TEST = 20
N_SENSORS = 21
N_OPERATIONAL_SETTINGS = 3
MAX_CYCLES = 350
MIN_CYCLES = 120
RANDOM_SEED = 42

# ── Feature Engineering ────────────────────────────────
ROLLING_WINDOWS = [5, 10, 20, 50]
RUL_CLIP = 125  # clip max RUL for piecewise-linear target

# ── Sensor columns (NASA C-MAPSS naming convention) ────
SENSOR_COLS = [f"sensor_{i}" for i in range(1, N_SENSORS + 1)]
SETTING_COLS = [f"setting_{i}" for i in range(1, N_OPERATIONAL_SETTINGS + 1)]

# ── Model Hyperparameters ──────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
}

LSTM_PARAMS = {
    "sequence_length": 30,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 50,
    "patience": 8,
}

ANOMALY_PARAMS = {
    "contamination": 0.05,
    "n_estimators": 200,
    "random_state": RANDOM_SEED,
}

# ── API ────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Dashboard ──────────────────────────────────────────
DASHBOARD_PORT = 8501

# ── Classification threshold (cycles until failure) ────
FAILURE_THRESHOLD = 30  # predict failure within 30 cycles
