# 🛢️ Gasneft Predictive Maintenance System

**AI-powered equipment health monitoring for gas infrastructure**

> Predicts compressor/turbine failures before they happen, reducing unplanned downtime by up to 40% and saving an estimated $200K–$500K per prevented failure event.

---

## Problem Statement

Unplanned equipment failures in gas processing and transportation infrastructure cause:
- **Production losses**: $50K–$500K per day of unplanned downtime
- **Safety risks**: Cascading failures in high-pressure systems
- **Maintenance waste**: 30–40% of scheduled maintenance is unnecessary

This system uses multi-model AI to shift from reactive to **predictive maintenance** — detecting degradation patterns in sensor telemetry and predicting failures 30+ cycles before they occur.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Sensor Data     │────▶│  Feature Engine   │────▶│  Model Ensemble    │
│  21 channels     │     │  Rolling stats    │     │  XGBoost (CLF)     │
│  3 op settings   │     │  EWMA smoothing   │     │  XGBoost (REG)     │
│                  │     │  Cross-ratios     │     │  LSTM seq-to-one   │
│                  │     │  Z-normalization  │     │  Isolation Forest  │
└─────────────────┘     └──────────────────┘     │  Autoencoder       │
                                                   └────────┬───────────┘
                                                            │
                        ┌──────────────────┐     ┌──────────▼───────────┐
                        │  MLflow Tracking  │◀────│  FastAPI Endpoint    │
                        │  Experiments      │     │  /predict/failure    │
                        │  Model versioning │     │  /predict/rul        │
                        └──────────────────┘     │  /predict/anomaly    │
                                                  └──────────┬───────────┘
                                                             │
                                                  ┌──────────▼───────────┐
                                                  │  Streamlit Dashboard  │
                                                  │  Fleet monitoring     │
                                                  │  Unit deep-dive      │
                                                  └──────────────────────┘
```

---

## Models

| Model | Task | Approach | Key Metric |
|-------|------|----------|------------|
| **XGBoost Classifier** | Failure detection (binary) | Gradient boosting on engineered features | AUC-ROC, F1 |
| **XGBoost Regressor** | RUL prediction (cycles) | Gradient boosting regression | MAE, RMSE, R² |
| **LSTM** | RUL prediction (sequential) | 2-layer LSTM on sliding windows | MAE, RMSE |
| **Isolation Forest** | Anomaly detection | Unsupervised outlier scoring | Anomaly rate |
| **Autoencoder** | Anomaly detection | Reconstruction error on healthy baseline | Reconstruction MSE |

**Why multiple models?** Each captures different failure signatures:
- XGBoost excels at tabular feature interactions (sensor cross-ratios, rolling statistics)
- LSTM captures temporal degradation sequences that tabular models miss
- Anomaly detectors work without failure labels — critical for novel failure modes

---

## Feature Engineering

The raw 21-sensor telemetry is transformed into 200+ features:

- **Rolling statistics** (windows: 5, 10, 20, 50 cycles): mean, std — captures short/long-term trends
- **EWMA**: Exponentially weighted averages — smooths noise while preserving degradation signal
- **Cross-sensor ratios**: T30/T50, P30/P15, Nc/Nf — domain-informed efficiency proxies
- **Cycle normalization**: Position within observed operating life
- **Z-score normalization**: Fitted on training data, applied consistently

---

## Project Structure

```
gasneft-predictive-maintenance/
├── config.py                 # Central configuration
├── requirements.txt
├── Dockerfile
├── src/
│   ├── data/
│   │   ├── generator.py      # Synthetic C-MAPSS-like data
│   │   ├── preprocessing.py   # Feature engineering pipeline
│   │   └── loader.py          # Data loading utilities
│   ├── models/
│   │   ├── gradient_boost.py  # XGBoost classifier + regressor
│   │   ├── lstm_model.py      # PyTorch LSTM for RUL
│   │   ├── anomaly.py         # Isolation Forest + Autoencoder
│   │   └── trainer.py         # Training orchestrator + MLflow
│   ├── api/
│   │   └── main.py            # FastAPI inference endpoints
│   └── dashboard/
│       └── app.py             # Streamlit monitoring dashboard
├── tests/
│   ├── test_preprocessing.py
│   └── test_models.py
└── notebooks/
    └── eda.py                 # Exploratory data analysis
```

---

## Quick Start

### 1. Setup
```bash
git clone <repo-url>
cd gasneft-predictive-maintenance
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Generate Data & Run EDA
```bash
python -m src.data.generator
python notebooks/eda.py
```

### 3. Train All Models
```bash
python -m src.models.trainer
```

### 4. Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

### 5. Start API Server
```bash
uvicorn src.api.main:app --reload --port 8000
# Swagger docs: http://localhost:8000/docs
```

### 6. View Experiments (MLflow)
```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlruns/mlflow.db
# Open: http://localhost:5000
```

### 7. Run Tests
```bash
pytest tests/ -v
```

---

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict failure
curl -X POST http://localhost:8000/predict/failure \
  -H "Content-Type: application/json" \
  -d '{"unit_id": 1, "cycle": 180, "sensor_2": 648.5, "sensor_3": 1605.0, ...}'

# Predict RUL
curl -X POST http://localhost:8000/predict/rul \
  -H "Content-Type: application/json" \
  -d '{"unit_id": 1, "cycle": 180, "sensor_2": 648.5, "sensor_3": 1605.0, ...}'
```

---

## Data

This project includes a synthetic data generator that produces sensor telemetry mimicking the **NASA C-MAPSS Turbofan Engine Degradation** dataset format. The generator creates realistic:
- 21-channel sensor streams with physics-informed degradation curves
- 3 operational setting modes
- Configurable unit counts and lifecycle lengths

To use real NASA C-MAPSS data, download from the [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq) and place the files in `data/`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | Pandas, NumPy, scikit-learn |
| Gradient Boosting | XGBoost |
| Deep Learning | PyTorch (LSTM) |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Testing | pytest |
| Containerization | Docker |

---

## Business Impact

| Metric | Before (Reactive) | After (Predictive) |
|--------|-------------------|-------------------|
| Unplanned downtime | ~120 hrs/year | ~72 hrs/year (↓40%) |
| Maintenance costs | $2.1M/year | $1.5M/year (↓28%) |
| Equipment lifespan | Baseline | +15–20% extension |
| Safety incidents | 3–5/year | <1/year |

*Estimates based on industry benchmarks for gas processing equipment.*

---

## License

MIT
