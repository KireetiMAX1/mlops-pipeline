"""
FastAPI model serving — loads the latest production model from the MLflow registry.

Endpoints:
  GET  /health               liveness check
  GET  /model/info           current model version and training metrics
  POST /predict              single prediction with optional SHAP explanation
  POST /predict/batch        batch predictions
  GET  /metrics/predictions  aggregated stats from the prediction log
  POST /model/reload         hot-reload model from registry

Run locally:
    uvicorn api.serve:app --reload --port 8000
"""

from __future__ import annotations
import os
import time
from typing import Optional

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "churn-predictor")
MODEL_STAGE           = os.getenv("MODEL_STAGE", "Production")

app = FastAPI(
    title="Churn Predictor API",
    description="Production model serving with MLflow registry, drift detection, and SHAP explanations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_info: dict = {}
_prediction_log: list[dict] = []


class CustomerFeatures(BaseModel):
    tenure: int            = Field(..., ge=0, le=120, description="Months as customer")
    monthly_charges: float = Field(..., ge=0, le=500)
    total_charges: float   = Field(..., ge=0)
    contract: int          = Field(..., ge=0, le=2, description="0=Month-to-month, 1=One year, 2=Two year")
    payment_method: int    = Field(..., ge=0, le=3)
    internet_service: int  = Field(..., ge=0, le=2)
    tech_support: int      = Field(..., ge=0, le=2)
    online_security: int   = Field(..., ge=0, le=2)

    @validator("total_charges")
    def total_gte_monthly(cls, v, values):
        if "monthly_charges" in values and v < values["monthly_charges"]:
            raise ValueError("total_charges must be >= monthly_charges")
        return v


class PredictionRequest(BaseModel):
    customer: CustomerFeatures
    return_explanation: bool = False


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures]


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_tier: str
    explanation: Optional[dict] = None
    model_version: str
    latency_ms: float


def get_risk_tier(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


def load_model():
    global _model, _model_info
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
        _model = mlflow.pyfunc.load_model(model_uri)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[MODEL_STAGE])
        if versions:
            v = versions[0]
            _model_info = {
                "name":    v.name,
                "version": v.version,
                "stage":   v.current_stage,
                "run_id":  v.run_id,
            }
            run = client.get_run(v.run_id)
            _model_info["metrics"] = {
                k: round(float(v2), 4)
                for k, v2 in run.data.metrics.items()
                if not k.startswith("cv_")
            }
        print(f"Model loaded: {REGISTERED_MODEL_NAME} ({MODEL_STAGE})")
    except Exception as e:
        print(f"Could not load from registry ({e}). Running in degraded mode.")
        _model = None


def features_to_df(customer: CustomerFeatures) -> pd.DataFrame:
    d = customer.dict()
    d["charges_per_tenure"]  = d["monthly_charges"] / (d["tenure"] + 1)
    d["high_value_customer"] = int(d["monthly_charges"] > 70)
    return pd.DataFrame([d])


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
def health():
    return {
        "status":        "ok" if _model else "degraded",
        "model_loaded":  _model is not None,
        "model_version": _model_info.get("version", "N/A"),
    }


@app.get("/model/info")
def model_info():
    if not _model_info:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model_info


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    if not _model:
        raise HTTPException(status_code=503, detail="Model not available")

    t0 = time.perf_counter()
    df = features_to_df(request.customer)
    prob = float(_model.predict(df)[0])
    latency = round((time.perf_counter() - t0) * 1000, 2)

    explanation = None
    if request.return_explanation:
        try:
            raw_model = _model._model_impl.python_model
            explainer = shap.TreeExplainer(raw_model)
            shap_vals = explainer.shap_values(df)
            explanation = {
                col: round(float(val), 4)
                for col, val in zip(df.columns, shap_vals[0])
            }
        except Exception:
            explanation = {"note": "SHAP unavailable for this model type"}

    record = {
        "timestamp":   time.time(),
        "features":    request.customer.dict(),
        "probability": prob,
        "prediction":  prob >= 0.5,
        "latency_ms":  latency,
    }
    background_tasks.add_task(_prediction_log.append, record)

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=prob >= 0.5,
        risk_tier=get_risk_tier(prob),
        explanation=explanation,
        model_version=_model_info.get("version", "unknown"),
        latency_ms=latency,
    )


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    if not _model:
        raise HTTPException(status_code=503, detail="Model not available")
    results = []
    for customer in request.customers:
        df = features_to_df(customer)
        prob = float(_model.predict(df)[0])
        results.append({
            "churn_probability": round(prob, 4),
            "churn_prediction":  prob >= 0.5,
            "risk_tier":         get_risk_tier(prob),
        })
    return {"predictions": results, "count": len(results)}


@app.get("/metrics/predictions")
def prediction_metrics():
    if not _prediction_log:
        return {"message": "No predictions logged yet"}
    probs = [r["probability"] for r in _prediction_log]
    latencies = [r["latency_ms"] for r in _prediction_log]
    return {
        "total_predictions":    len(_prediction_log),
        "avg_churn_prob":       round(float(np.mean(probs)), 4),
        "churn_rate_predicted": round(float(np.mean([p >= 0.5 for p in probs])), 4),
        "avg_latency_ms":       round(float(np.mean(latencies)), 2),
        "p99_latency_ms":       round(float(np.percentile(latencies, 99)), 2),
    }


@app.post("/model/reload")
def reload_model():
    load_model()
    return {"status": "reloaded", "model_info": _model_info}


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
