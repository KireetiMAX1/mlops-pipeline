# MLOps Pipeline — End-to-End ML Platform

A production-grade MLOps system for customer churn prediction covering the complete lifecycle: data versioning, experiment tracking, model registry, REST API serving, data drift detection, automated retraining, and a live monitoring dashboard.

[![CI/CD](https://github.com/kireetiaddagada/mlops-pipeline/actions/workflows/mlops_pipeline.yml/badge.svg)](https://github.com/kireetiaddagada/mlops-pipeline/actions)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-Versioned-945DD6)](https://dvc.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
                   GitHub Actions CI/CD
                   lint -> validate -> train -> deploy
                   + scheduled weekly retraining
                              |
          ┌───────────────────┼──────────────────────┐
          │                   │                      │
   data/              training/              monitoring/
   data_pipeline.py   train.py               drift_detector.py

   - Schema check     - 4 model types         - Evidently AI
   - Feature eng      - 5-fold CV             - PSI / KS tests
   - DVC versioning   - SHAP plots            - HTML reports
                      - MLflow tracking       - Auto-trigger
                              │
                       Model Registry (MLflow)
                       None -> Staging -> Production
                              │
                       api/serve.py
                       FastAPI + Docker
                       POST /predict
                       POST /predict/batch
                       GET  /model/info
                       GET  /metrics/predictions
                              │
                       dashboard/app.py (Streamlit)
                       - MLflow run history
                       - Model registry view
                       - Drift report viewer
                       - Live prediction sandbox
```

---

## Features

- **Data versioning** — DVC tracks datasets like source code; every training run is fully reproducible
- **Experiment tracking** — MLflow logs hyperparameters, metrics, SHAP plots, and artifacts per run
- **Model registry** — promote models through None > Staging > Production with a single MLflow command
- **4-model comparison** — XGBoost, RandomForest, GradientBoosting, LogisticRegression
- **SHAP explainability** — feature importance plots automatically logged to MLflow
- **FastAPI serving** — `/predict` with SHAP explanation, `/predict/batch`, `/model/info`
- **Drift detection** — Evidently AI detects feature and target distribution shift; results logged to MLflow
- **Performance gate** — CI fails if ROC-AUC drops below 0.75, blocking bad models from the registry
- **Automated retraining** — GitHub Actions triggers on a weekly schedule or when drift is detected
- **Dockerized API** — production-ready container with health check endpoint

---

## Model Results

| Model              | ROC-AUC   | F1   | Precision | Recall |
|--------------------|-----------|------|-----------|--------|
| XGBoost            | **0.892** | 0.71 | 0.74      | 0.69   |
| GradientBoosting   | 0.881     | 0.70 | 0.72      | 0.68   |
| RandomForest       | 0.874     | 0.68 | 0.73      | 0.64   |
| LogisticRegression | 0.821     | 0.62 | 0.67      | 0.58   |

---

## Quickstart

```bash
git clone https://github.com/kireetiaddagada/mlops-pipeline
cd mlops-pipeline

pip install -r requirements-dev.txt

# 1. Generate and validate data
python data/data_pipeline.py

# 2. Train a model (tracked in MLflow)
python training/train.py --experiment my-experiment --model xgboost

# 3. Open MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Visit http://localhost:5000

# 4. Serve the API
uvicorn api.serve:app --reload --port 8000
# Visit http://localhost:8000/docs

# 5. Run drift detection
python monitoring/drift_detector.py --drift_factor 0.5

# 6. Open the dashboard
streamlit run dashboard/app.py

# 7. Run tests
pytest tests/ -v --cov
```

---

## CI/CD Pipeline

Every push to `main` triggers:

```
push to main
    |
    ├── Lint (ruff) + Type check (mypy)
    ├── Tests (pytest --cov)
    ├── Data schema validation
    ├── Drift detection report
    └── Docker build + health check smoke test
         |
         └── [on schedule / [retrain] commit] -> Train -> Performance gate -> Registry
```

Trigger retraining manually by including `[retrain]` in your commit message, or via `workflow_dispatch` with `force_retrain=true`.

---

## API Reference

```bash
# Health check
curl http://localhost:8000/health

# Single prediction with SHAP explanation
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer": {
      "tenure": 12,
      "monthly_charges": 85.0,
      "total_charges": 1020.0,
      "contract": 0,
      "payment_method": 0,
      "internet_service": 0,
      "tech_support": 1,
      "online_security": 1
    },
    "return_explanation": true
  }'
```

Example response:
```json
{
  "churn_probability": 0.7823,
  "churn_prediction": true,
  "risk_tier": "HIGH",
  "explanation": {"tenure": -0.12, "monthly_charges": 0.31},
  "model_version": "3",
  "latency_ms": 4.2
}
```

---

## Project Structure

```
mlops-pipeline/
├── data/
│   ├── data_pipeline.py       ingestion, schema validation, preprocessing
│   ├── raw/                   raw datasets (versioned with DVC)
│   └── processed/             feature-engineered datasets (versioned with DVC)
├── training/
│   └── train.py               multi-model training with MLflow tracking
├── api/
│   └── serve.py               FastAPI serving layer
├── monitoring/
│   ├── drift_detector.py      Evidently drift detection + MLflow logging
│   └── reports/               generated HTML drift reports
├── dashboard/
│   └── app.py                 Streamlit monitoring dashboard
├── tests/
│   └── test_pipeline.py       unit and integration tests
├── infra/
│   └── Dockerfile             production container
├── .github/workflows/
│   └── mlops_pipeline.yml     CI/CD + scheduled retraining
├── dvc.yaml                   reproducible pipeline stages
├── requirements.txt           runtime dependencies
└── requirements-dev.txt       development and testing dependencies
```

---

## Configuration

Copy `.env.example` to `.env` and set values as needed:

```
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
REGISTERED_MODEL_NAME=churn-predictor
MODEL_STAGE=Production
API_BASE_URL=http://localhost:8000
DRIFT_THRESHOLD=0.15
```

For production, swap the SQLite MLflow backend for PostgreSQL and configure a DVC remote (S3, GCS, or Azure Blob).

---

## License

MIT © Kireeti Addagada
