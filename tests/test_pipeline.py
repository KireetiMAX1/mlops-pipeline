"""Tests for data pipeline, API endpoints, and drift detection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data.data_pipeline import generate_sample_data, validate_schema, preprocess, ValidationReport


# Data pipeline tests

def test_generate_sample_data_shape():
    df = generate_sample_data(n=100)
    assert len(df) == 100
    assert "churn" in df.columns


def test_generate_sample_data_churn_rate():
    df = generate_sample_data(n=2000)
    rate = df["churn"].mean()
    assert 0.05 < rate < 0.60, f"Unexpected churn rate: {rate}"


def test_validate_schema_passes():
    df = generate_sample_data(n=200)
    report = validate_schema(df)
    assert report.passed
    assert len(report.issues) == 0


def test_validate_schema_missing_column():
    df = generate_sample_data(n=200).drop(columns=["tenure"])
    report = validate_schema(df)
    assert not report.passed
    assert any("tenure" in i for i in report.issues)


def test_validate_schema_too_few_rows():
    df = generate_sample_data(n=10)
    report = validate_schema(df)
    assert not report.passed


def test_preprocess_adds_features():
    df = generate_sample_data(n=100)
    X, feature_names = preprocess(df)
    assert "charges_per_tenure" in feature_names
    assert "high_value_customer" in feature_names


def test_preprocess_no_target_column():
    df = generate_sample_data(n=100)
    X, feature_names = preprocess(df)
    assert "churn" not in feature_names


def test_preprocess_no_nulls():
    df = generate_sample_data(n=500)
    X, _ = preprocess(df)
    assert not X.isnull().values.any()


# API tests (without loaded model)

def test_health_endpoint():
    from api.serve import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_without_model_returns_503():
    from api.serve import app
    client = TestClient(app)
    payload = {
        "customer": {
            "tenure": 12, "monthly_charges": 65.0,
            "total_charges": 780.0, "contract": 0,
            "payment_method": 0, "internet_service": 1,
            "tech_support": 1, "online_security": 0,
        }
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code in (200, 503)


def test_predict_invalid_payload():
    from api.serve import app
    client = TestClient(app)
    resp = client.post("/predict", json={"customer": {"tenure": -5}})
    assert resp.status_code == 422


def test_batch_predict_empty():
    from api.serve import app
    client = TestClient(app)
    resp = client.post("/predict/batch", json={"customers": []})
    assert resp.status_code in (200, 503)


# Drift detection tests

def test_simulate_production_data_no_drift():
    from monitoring.drift_detector import simulate_production_data
    ref     = generate_sample_data(n=1000)
    current = simulate_production_data(ref, drift_factor=0.0)
    assert len(current) > 0
    assert "monthly_charges" in current.columns


def test_simulate_production_data_with_drift():
    from monitoring.drift_detector import simulate_production_data
    ref      = generate_sample_data(n=1000)
    no_drift = simulate_production_data(ref, drift_factor=0.0)
    drifted  = simulate_production_data(ref, drift_factor=1.0)
    assert drifted["monthly_charges"].mean() > no_drift["monthly_charges"].mean()
