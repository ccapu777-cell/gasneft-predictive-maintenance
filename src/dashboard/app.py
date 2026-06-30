"""
Gasneft Predictive Maintenance Dashboard
==========================================
Interactive Streamlit dashboard for equipment health monitoring.

Sections:
  1. Fleet Overview — all units health status
  2. Unit Deep-Dive — sensor trends, RUL prediction, anomaly scores
  3. Model Performance — training metrics comparison
  4. Feature Importance — what drives predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.data.loader import load_train_test, get_feature_columns
from src.data.preprocessing import build_features, add_rul_column, add_failure_label

# ═══════════════════════════════════════════════════════
#  Page Config
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="Gasneft Predictive Maintenance",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B3A4B;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .status-critical { color: #FF4B4B; font-weight: bold; }
    .status-warning { color: #FFA500; font-weight: bold; }
    .status-healthy { color: #00CC66; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  Data Loading (cached)
# ═══════════════════════════════════════════════════════

@st.cache_data
def load_data():
    train_df, test_df, rul_true = load_train_test()
    train_df = add_rul_column(train_df)
    train_df = add_failure_label(train_df)
    return train_df, test_df, rul_true


@st.cache_data
def load_predictions():
    """Load pre-computed predictions if available."""
    model_dir = Path(config.MODELS_DIR)
    preds = {}
    try:
        clf = joblib.load(model_dir / "xgb_classifier.joblib")
        reg = joblib.load(model_dir / "xgb_regressor.joblib")
        scaler = joblib.load(model_dir / "scaler.joblib")
        preds = {"classifier": clf, "regressor": reg, "scaler": scaler}
    except Exception:
        pass
    return preds


@st.cache_data
def load_feature_importance():
    fi_path = Path(config.MODELS_DIR) / "feature_importance_clf.csv"
    if fi_path.exists():
        return pd.read_csv(fi_path)
    return None


# ═══════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════

st.sidebar.markdown("## 🛢️ Gasneft PdM")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Fleet Overview", "🔍 Unit Deep-Dive", "📊 Model Performance", "⚙️ Feature Importance"],
)

train_df, test_df, rul_true = load_data()
preds = load_predictions()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Training units:** {train_df['unit_id'].nunique()}")
st.sidebar.markdown(f"**Total records:** {len(train_df):,}")
st.sidebar.markdown(f"**Sensors:** {config.N_SENSORS}")


# ═══════════════════════════════════════════════════════
#  Page 1: Fleet Overview
# ═══════════════════════════════════════════════════════

if page == "🏠 Fleet Overview":
    st.markdown('<p class="main-header">Fleet Health Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time equipment monitoring across all units</p>', unsafe_allow_html=True)
    st.markdown("")

    # ── KPI Cards ──────────────────────────────────────
    unit_summary = train_df.groupby("unit_id").agg(
        max_cycle=("cycle", "max"),
        min_rul=("RUL", "min"),
        mean_rul=("RUL", "mean"),
        has_failure=("label", "max"),
    ).reset_index()

    critical = (unit_summary["min_rul"] <= 30).sum()
    warning = ((unit_summary["min_rul"] > 30) & (unit_summary["min_rul"] <= 60)).sum()
    healthy = (unit_summary["min_rul"] > 60).sum()
    total = len(unit_summary)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Units", total)
    c2.metric("🟢 Healthy", healthy)
    c3.metric("🟡 Warning", warning)
    c4.metric("🔴 Critical", critical)

    st.markdown("---")

    # ── Fleet RUL Distribution ─────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Unit Lifecycle Distribution")
        fig = px.histogram(
            unit_summary, x="max_cycle", nbins=20,
            color_discrete_sequence=["#667eea"],
            labels={"max_cycle": "Total Operational Cycles"},
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Health Status Breakdown")
        status_data = pd.DataFrame({
            "Status": ["Healthy (>60)", "Warning (30-60)", "Critical (<30)"],
            "Count": [healthy, warning, critical],
        })
        fig = px.pie(
            status_data, values="Count", names="Status",
            color="Status",
            color_discrete_map={
                "Healthy (>60)": "#00CC66",
                "Warning (30-60)": "#FFA500",
                "Critical (<30)": "#FF4B4B",
            },
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap: sensor values across units ────────────
    st.subheader("Sensor Health Heatmap (Last Cycle per Unit)")
    last_readings = train_df.groupby("unit_id").tail(1)[["unit_id"] + config.SENSOR_COLS[:12]]
    last_readings = last_readings.set_index("unit_id")

    # Normalize for heatmap
    from sklearn.preprocessing import MinMaxScaler
    scaler_viz = MinMaxScaler()
    normalized = pd.DataFrame(
        scaler_viz.fit_transform(last_readings),
        index=last_readings.index,
        columns=last_readings.columns,
    )

    fig = px.imshow(
        normalized.T,
        labels=dict(x="Unit ID", y="Sensor", color="Normalized Value"),
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
#  Page 2: Unit Deep-Dive
# ═══════════════════════════════════════════════════════

elif page == "🔍 Unit Deep-Dive":
    st.markdown('<p class="main-header">Unit Deep-Dive Analysis</p>', unsafe_allow_html=True)

    unit_ids = sorted(train_df["unit_id"].unique())
    selected_unit = st.sidebar.selectbox("Select Unit", unit_ids)

    unit_data = train_df[train_df["unit_id"] == selected_unit].sort_values("cycle")

    # ── Unit KPIs ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cycles", int(unit_data["cycle"].max()))
    c2.metric("Current RUL", int(unit_data["RUL"].iloc[-1]))
    min_rul = int(unit_data["RUL"].min())
    status = "CRITICAL" if min_rul <= 30 else "WARNING" if min_rul <= 60 else "HEALTHY"
    c3.metric("Status", status)
    c4.metric("Failure Predicted", "Yes" if unit_data["label"].iloc[-1] == 1 else "No")

    st.markdown("---")

    # ── Sensor Trends ──────────────────────────────────
    st.subheader("Sensor Trends Over Time")

    sensor_select = st.multiselect(
        "Select sensors to display",
        config.SENSOR_COLS,
        default=["sensor_2", "sensor_3", "sensor_7", "sensor_11", "sensor_15"],
    )

    if sensor_select:
        fig = make_subplots(rows=len(sensor_select), cols=1, shared_xaxes=True,
                            subplot_titles=sensor_select, vertical_spacing=0.04)
        colors = px.colors.qualitative.Set2

        for i, sensor in enumerate(sensor_select):
            fig.add_trace(
                go.Scatter(
                    x=unit_data["cycle"], y=unit_data[sensor],
                    mode="lines", name=sensor,
                    line=dict(color=colors[i % len(colors)], width=1.5),
                ),
                row=i + 1, col=1,
            )
            # Mark failure zone
            if unit_data["RUL"].min() <= config.FAILURE_THRESHOLD:
                fail_start = unit_data[unit_data["RUL"] <= config.FAILURE_THRESHOLD]["cycle"].min()
                fig.add_vrect(
                    x0=fail_start, x1=unit_data["cycle"].max(),
                    fillcolor="red", opacity=0.1, line_width=0,
                    row=i + 1, col=1,
                )

        fig.update_layout(height=200 * len(sensor_select), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── RUL Curve ──────────────────────────────────────
    st.subheader("Remaining Useful Life Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=unit_data["cycle"], y=unit_data["RUL"],
        mode="lines", name="True RUL",
        line=dict(color="#667eea", width=2),
        fill="tozeroy", fillcolor="rgba(102,126,234,0.1)",
    ))
    fig.add_hline(y=config.FAILURE_THRESHOLD, line_dash="dash",
                  line_color="red", annotation_text="Failure Threshold")
    fig.update_layout(
        xaxis_title="Cycle", yaxis_title="RUL (cycles)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
#  Page 3: Model Performance
# ═══════════════════════════════════════════════════════

elif page == "📊 Model Performance":
    st.markdown('<p class="main-header">Model Performance Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Side-by-side evaluation of all trained models</p>', unsafe_allow_html=True)

    # ── Model comparison table ─────────────────────────
    st.subheader("Model Metrics Summary")

    metrics_data = {
        "Model": [
            "XGBoost Classifier", "XGBoost Regressor",
            "LSTM (RUL)", "Isolation Forest", "Autoencoder",
        ],
        "Task": [
            "Failure Detection", "RUL Prediction",
            "RUL Prediction", "Anomaly Detection", "Anomaly Detection",
        ],
        "Primary Metric": ["AUC-ROC", "MAE", "MAE", "F1-Score", "F1-Score"],
        "Architecture": [
            "Gradient Boosting (300 trees)", "Gradient Boosting (300 trees)",
            "2-layer LSTM (64 hidden)", "200 estimators", "64→32→16→32→64",
        ],
        "Key Strength": [
            "Fast inference, interpretable",
            "Handles tabular features well",
            "Captures temporal degradation patterns",
            "No labels needed, fast training",
            "Learns normal operating envelope",
        ],
    }

    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── RUL Distribution Analysis ──────────────────────
    st.subheader("RUL Distribution Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            train_df, x="RUL", nbins=50,
            color_discrete_sequence=["#764ba2"],
            title="RUL Distribution (Training Data)",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        label_dist = train_df["label"].value_counts().reset_index()
        label_dist.columns = ["Class", "Count"]
        label_dist["Class"] = label_dist["Class"].map({0: "Healthy", 1: "Failure"})
        fig = px.bar(
            label_dist, x="Class", y="Count",
            color="Class",
            color_discrete_map={"Healthy": "#00CC66", "Failure": "#FF4B4B"},
            title="Class Distribution (Failure Detection)",
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Architecture Diagram ───────────────────────────
    st.subheader("System Architecture")
    st.markdown("""
    ```
    ┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
    │  Sensor Data     │────▶│  Feature Engine   │────▶│  Model Ensemble    │
    │  (21 channels)   │     │  • Rolling stats  │     │  • XGBoost (CLF)   │
    │  + 3 settings    │     │  • EWMA           │     │  • XGBoost (REG)   │
    │                  │     │  • Cross-ratios   │     │  • LSTM            │
    │                  │     │  • Normalization  │     │  • Isolation Forest │
    └─────────────────┘     └──────────────────┘     │  • Autoencoder     │
                                                       └────────┬───────────┘
                                                                │
                            ┌──────────────────┐     ┌──────────▼───────────┐
                            │  MLflow Tracking  │◀────│  Prediction API      │
                            │  • Experiments    │     │  /predict/failure    │
                            │  • Model registry │     │  /predict/rul        │
                            │  • Metrics        │     │  /predict/anomaly    │
                            └──────────────────┘     └──────────┬───────────┘
                                                                │
                                                     ┌──────────▼───────────┐
                                                     │  Dashboard           │
                                                     │  • Fleet overview    │
                                                     │  • Unit deep-dive    │
                                                     │  • Model comparison  │
                                                     └──────────────────────┘
    ```
    """)


# ═══════════════════════════════════════════════════════
#  Page 4: Feature Importance
# ═══════════════════════════════════════════════════════

elif page == "⚙️ Feature Importance":
    st.markdown('<p class="main-header">Feature Importance Analysis</p>', unsafe_allow_html=True)

    fi = load_feature_importance()

    if fi is not None:
        st.subheader("Top 20 Features — XGBoost Classifier")
        fig = px.bar(
            fi.head(20).sort_values("importance"),
            x="importance", y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=600, yaxis_title="", xaxis_title="Importance Score")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Categories Breakdown")

        fi["category"] = fi["feature"].apply(lambda x:
            "Rolling Stats" if "rmean" in x or "rstd" in x
            else "EWMA" if "ewma" in x
            else "Cross-Ratios" if "ratio" in x
            else "Raw Sensors" if "sensor" in x
            else "Settings" if "setting" in x
            else "Other"
        )
        cat_importance = fi.groupby("category")["importance"].sum().reset_index()
        fig = px.pie(cat_importance, values="importance", names="category",
                     title="Importance by Feature Category")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Feature importance data not available. Run training first.")
        st.code("python -m src.models.trainer", language="bash")

    # ── Sensor Correlation Matrix ──────────────────────
    st.subheader("Sensor Correlation Matrix")
    corr = train_df[config.SENSOR_COLS].corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Sensor-to-Sensor Correlations",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
#  Footer
# ═══════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown("**Gasneft PdM v1.0**")
st.sidebar.markdown("Built for AI Engineering showcase")
