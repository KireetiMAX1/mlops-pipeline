"""
MLOps Monitoring Dashboard built with Streamlit.

Displays:
  - Model performance history from MLflow
  - Data drift status from Evidently reports
  - Live prediction stats from the FastAPI service
  - Model registry versions and promotion status

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
API_BASE_URL        = os.getenv("API_BASE_URL", "http://localhost:8000")
REPORTS_DIR         = Path("monitoring/reports")

st.set_page_config(
    page_title="MLOps Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()


@st.cache_data(ttl=30)
def get_experiment_runs(experiment_name: str = "churn-prediction") -> pd.DataFrame:
    try:
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=50,
        )
        rows = []
        for r in runs:
            rows.append({
                "run_id":    r.info.run_id[:8],
                "model":     r.data.params.get("model_type", "unknown"),
                "roc_auc":   r.data.metrics.get("roc_auc", 0),
                "f1":        r.data.metrics.get("f1", 0),
                "precision": r.data.metrics.get("precision", 0),
                "recall":    r.data.metrics.get("recall", 0),
                "started":   datetime.fromtimestamp(r.info.start_time / 1000),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def get_model_versions() -> list[dict]:
    try:
        versions = client.search_model_versions("name='churn-predictor'")
        return [
            {
                "version": v.version,
                "stage":   v.current_stage,
                "run_id":  v.run_id[:8],
                "created": datetime.fromtimestamp(v.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M"),
            }
            for v in versions
        ]
    except Exception:
        return []


@st.cache_data(ttl=10)
def get_api_metrics() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/metrics/predictions", timeout=3)
        return r.json() if r.ok else {}
    except Exception:
        return {}


@st.cache_data(ttl=10)
def get_api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.json() if r.ok else {"status": "unreachable"}
    except Exception:
        return {"status": "unreachable"}


def get_latest_drift_summary() -> dict | None:
    summaries = sorted(REPORTS_DIR.glob("drift_summary_*.json"), reverse=True)
    if summaries:
        return json.loads(summaries[0].read_text())
    return None


# Header
st.title("MLOps Monitoring Dashboard")
st.caption(f"Auto-refreshes every 30s  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

# API Health
health = get_api_health()
status_color = "green" if health.get("status") == "ok" else "orange" if health.get("status") == "degraded" else "red"
st.markdown(
    f"**API Status**: `{health.get('status', 'unknown')}` | "
    f"Model version: `{health.get('model_version', 'N/A')}`"
)

st.divider()

# Top metrics
col1, col2, col3, col4 = st.columns(4)
runs_df       = get_experiment_runs()
api_metrics   = get_api_metrics()
drift_summary = get_latest_drift_summary()

with col1:
    if not runs_df.empty:
        st.metric("Best ROC-AUC", f"{runs_df['roc_auc'].max():.4f}")
    else:
        st.metric("Best ROC-AUC", "No runs yet")

with col2:
    st.metric("Total Predictions", api_metrics.get("total_predictions", "—"))

with col3:
    avg_latency = api_metrics.get("avg_latency_ms")
    st.metric("Avg Latency", f"{avg_latency:.1f}ms" if avg_latency else "—")

with col4:
    if drift_summary:
        drift_status = "DRIFT DETECTED" if drift_summary["drift_detected"] else "Stable"
        st.metric("Data Drift", drift_status)
    else:
        st.metric("Data Drift", "No report yet")

st.divider()

# Experiment runs chart
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Model Performance — All Runs")
    if not runs_df.empty:
        fig = px.scatter(
            runs_df, x="started", y="roc_auc", color="model",
            size="f1", hover_data=["run_id", "precision", "recall"],
            title="ROC-AUC over time by model type",
            labels={"roc_auc": "ROC-AUC", "started": "Run date"},
        )
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            runs_df.sort_values("roc_auc", ascending=False).head(10),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No MLflow runs found. Run `python training/train.py` to start tracking.")

with col_right:
    st.subheader("Model Registry")
    versions = get_model_versions()
    if versions:
        for v in versions:
            badge = {"Production": "[PROD]", "Staging": "[STG]", "None": "[NONE]"}.get(v["stage"], "")
            st.markdown(
                f"{badge} **v{v['version']}** — {v['stage']}\n\n"
                f"  Run: `{v['run_id']}` · {v['created']}"
            )
    else:
        st.info("No registered models yet.")

    st.divider()
    st.subheader("Drift Report")
    if drift_summary:
        st.json({k: v for k, v in drift_summary.items() if k != "report_path"})
        html_path = Path(drift_summary.get("report_path", ""))
        if html_path.exists():
            st.download_button(
                "Download Full Drift Report",
                data=html_path.read_text(),
                file_name=html_path.name,
                mime="text/html",
            )
    else:
        st.info("Run `python monitoring/drift_detector.py` to generate a drift report.")

# Live prediction simulator
st.divider()
st.subheader("Live Prediction Simulator")
with st.expander("Try the model API"):
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure  = st.slider("Tenure (months)", 1, 72, 12)
        monthly = st.slider("Monthly charges ($)", 20, 120, 65)
    with c2:
        total    = st.number_input("Total charges ($)", value=float(tenure * monthly))
        contract = st.selectbox("Contract", [0, 1, 2],
                                format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])
    with c3:
        internet = st.selectbox("Internet", [0, 1, 2],
                                format_func=lambda x: ["Fiber optic", "DSL", "None"][x])
        tech     = st.selectbox("Tech support", [0, 1, 2],
                                format_func=lambda x: ["Yes", "No", "No internet"][x])
        security = st.selectbox("Online security", [0, 1, 2],
                                format_func=lambda x: ["Yes", "No", "No internet"][x])

    if st.button("Get Prediction"):
        payload = {
            "customer": {
                "tenure": tenure, "monthly_charges": monthly,
                "total_charges": total, "contract": contract,
                "payment_method": 0, "internet_service": internet,
                "tech_support": tech, "online_security": security,
            },
            "return_explanation": True,
        }
        try:
            resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
            if resp.ok:
                result = resp.json()
                risk_colors = {"HIGH": "HIGH RISK", "MEDIUM": "MEDIUM RISK", "LOW": "LOW RISK"}
                st.success(
                    f"Churn probability: **{result['churn_probability']:.1%}** | "
                    f"Risk: **{result['risk_tier']}** | "
                    f"Latency: {result['latency_ms']}ms"
                )
                if result.get("explanation"):
                    st.json(result["explanation"])
            else:
                st.error(f"API error: {resp.status_code}")
        except Exception as e:
            st.warning(f"API not reachable ({e}). Start with: `uvicorn api.serve:app --port 8000`")
