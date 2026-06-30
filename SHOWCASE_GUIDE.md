# Showcase Guide — Gasneft Predictive Maintenance

How to set up and demo this project step by step.

---

## Prerequisites

Make sure you have these installed:
- Python 3.10+ (3.11 recommended)
- pip
- Git
- A terminal (any OS works, Linux/Mac preferred)

---

## Step 1 — Clone and Set Up Environment

```bash
# If you have the zip, unzip it. Otherwise:
git clone <your-repo-url>
cd gasneft-predictive-maintenance

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

**Expected time:** ~2 minutes

---

## Step 2 — Generate the Dataset

```bash
python -m src.data.generator
```

**What happens:** Creates 80 training engine units and 20 test units with simulated sensor degradation data. Output goes to `data/`.

**What to say in the demo:** "I built a synthetic data generator that mirrors NASA's C-MAPSS turbofan degradation format — 21 sensor channels with physics-informed degradation curves. The system is designed to also accept real sensor data from production equipment."

---

## Step 3 — Run Exploratory Data Analysis

```bash
python notebooks/eda.py
```

**What happens:** Generates 6 publication-quality plots in `data/eda_plots/`:
1. Unit lifetime distribution
2. Sensor value distributions (all 21 channels)
3. Degradation trends over time
4. Sensor correlation matrix
5. RUL distribution + class balance
6. Sensor vs RUL scatter plots

**What to show:** Open the plots in `data/eda_plots/`. Walk through 2-3 of them. The degradation trend plot (03) is the most visually compelling — it shows how sensors drift as engines approach failure.

---

## Step 4 — Run Tests

```bash
pytest tests/ -v
```

**What happens:** Runs 28 tests covering data generation, preprocessing, and model training.

**What to say:** "Before training, I validate the entire pipeline. 28 tests cover data integrity, feature engineering correctness, and model training sanity. All green."

---

## Step 5 — Train All Models

```bash
python -m src.models.trainer
```

**What happens:** Trains 5 models sequentially:
1. XGBoost Classifier (failure detection) — ~10 seconds
2. XGBoost Regressor (RUL prediction) — ~10 seconds
3. LSTM neural network (sequential RUL) — ~30-60 seconds
4. Isolation Forest (anomaly detection) — ~5 seconds
5. Autoencoder (anomaly detection) — ~15 seconds

All experiments are tracked in MLflow.

**Expected results:**
- XGBoost Classifier: AUC ~0.99, F1 ~0.98
- XGBoost Regressor: MAE ~2-3 cycles, R² ~0.98
- LSTM: MAE ~8-10 cycles
- Isolation Forest: F1 ~0.49
- Autoencoder: F1 ~0.59

**What to say:** "I trained 5 different model architectures because each captures different failure signatures. XGBoost dominates on tabular features — cross-sensor ratios, rolling statistics. The LSTM captures temporal degradation sequences that gradient boosting can't see. Anomaly detectors work without labels, which is critical for novel failure modes that aren't in historical data."

---

## Step 6 — Launch the Dashboard

```bash
streamlit run src/dashboard/app.py
```

**What happens:** Opens an interactive dashboard at http://localhost:8501

**Demo walkthrough:**

1. **Fleet Overview** (default page)
   - Show the KPI cards: total units, healthy/warning/critical counts
   - Point at the health status pie chart
   - Show the sensor heatmap — "Each column is a unit, each row is a sensor. Red zones indicate degradation."

2. **Unit Deep-Dive** (select from sidebar)
   - Pick a unit with high cycle count
   - Show sensor trend lines with the red failure zone overlay
   - Show the RUL curve declining to the threshold line

3. **Model Performance**
   - Walk through the comparison table
   - Show the architecture diagram
   - Point out the class distribution chart (show you understand imbalanced data)

4. **Feature Importance**
   - Show top 20 features bar chart
   - Show the feature category pie chart — "Rolling statistics dominate, which makes physical sense — degradation is a trend, not a single reading."
   - Show sensor correlation matrix

---

## Step 7 — Start the API Server

Open a second terminal:

```bash
cd gasneft-predictive-maintenance
source venv/bin/activate
uvicorn src.api.main:app --reload --port 8000
```

**Demo the Swagger docs:** Open http://localhost:8000/docs

Show a live API call:

```bash
# In a third terminal:
curl -X POST http://localhost:8000/predict/failure \
  -H "Content-Type: application/json" \
  -d '{
    "unit_id": 1,
    "cycle": 250,
    "sensor_2": 650.5,
    "sensor_3": 1610.0,
    "sensor_4": 1420.0,
    "sensor_7": 560.0,
    "sensor_11": 48.5,
    "sensor_15": 8.7
  }'
```

**What to say:** "The FastAPI endpoint serves predictions in real-time. In production, this would receive sensor telemetry from SCADA systems. The Swagger docs auto-generate from Pydantic schemas — any engineer can test the API without reading the code."

---

## Step 8 — Show MLflow Experiments

```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlruns/mlflow.db --port 5000
```

Open http://localhost:5000

**What to show:** Click into the experiment. Show how each model run tracks parameters, metrics, and model artifacts. Compare runs side by side.

**What to say:** "Every experiment is versioned and reproducible. I can compare hyperparameter configurations, roll back to any previous model, and track metric trends across iterations."

---

## Demo Script (5-minute version)

If time is limited, this is the fastest impactful path:

1. Open README.md → show architecture diagram and business impact table (30 sec)
2. `pytest tests/ -v` → all green (30 sec)
3. `python -m src.models.trainer` → watch models train, point out metrics (1 min)
4. `streamlit run src/dashboard/app.py` → Fleet Overview → Unit Deep-Dive → Feature Importance (2 min)
5. `uvicorn src.api.main:app` → hit `/docs`, send one prediction (1 min)

---

## Talking Points for Q&A

**"Why not just one model?"**
Different architectures capture different failure patterns. XGBoost excels at cross-sensor interactions from engineered features. LSTM captures temporal sequences — the trajectory matters, not just the snapshot. Anomaly detectors handle novel failure modes with zero labels.

**"How would this scale to production?"**
The FastAPI layer is already async. For fleet-scale deployment: Kafka ingestion, model serving via TorchServe/Triton, time-series database (InfluxDB/TimescaleDB), and alerting via Grafana thresholds.

**"Why synthetic data?"**
The generator mirrors NASA C-MAPSS format — the industry-standard benchmark for predictive maintenance research. The architecture works identically with real sensor data; just swap the CSV source.

**"What about the anomaly detectors' lower F1?"**
That's expected and actually a feature. They're unsupervised — they don't need failure labels. In practice, they catch failure modes that aren't in historical training data. The XGBoost handles known patterns; anomaly detectors handle unknowns.

**"What would you improve next?"**
Transformer architecture for the sequential model (attention over sensor channels), online learning for model drift, and a feedback loop where operator confirmations retrain the classifier.
