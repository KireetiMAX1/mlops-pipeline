"""
Training pipeline with MLflow experiment tracking and model registry.

Trains one of four classifiers on the churn dataset, logs all metrics,
SHAP plots, and registers the best model in MLflow.

Run:
    python training/train.py --experiment churn-v1 --model xgboost
    python training/train.py --experiment churn-v1 --model random_forest
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from data.data_pipeline import load_and_preprocess

MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
ARTIFACT_PATH         = "model"
REGISTERED_MODEL_NAME = "churn-predictor"


MODELS = {
    "xgboost": xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
    ),
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
}


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":      round(accuracy_score(y_true, y_pred), 4),
        "f1":            round(f1_score(y_true, y_pred), 4),
        "precision":     round(precision_score(y_true, y_pred), 4),
        "recall":        round(recall_score(y_true, y_pred), 4),
        "roc_auc":       round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
    }


def log_shap_plot(model, X_val: pd.DataFrame, run_id: str):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val.head(200))
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_val.head(200), show=False)
        path = f"/tmp/shap_summary_{run_id}.png"
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        mlflow.log_artifact(path, "plots")
    except Exception as e:
        print(f"  SHAP plot skipped: {e}")


def train(experiment_name: str, model_name: str, data_path: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    print(f"\nTraining: {model_name} | Experiment: {experiment_name}")
    X_train, X_val, y_train, y_val, feature_names = load_and_preprocess(data_path)

    model = MODELS[model_name]

    with mlflow.start_run(run_name=f"{model_name}-run") as run:
        run_id = run.info.run_id
        print(f"  MLflow run ID: {run_id}")

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("data_path", data_path)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        mlflow.log_metric("cv_roc_auc_mean", round(cv_scores.mean(), 4))
        mlflow.log_metric("cv_roc_auc_std",  round(cv_scores.std(), 4))
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(y_val, y_pred, y_prob)
        mlflow.log_metrics(metrics)
        print(f"  Validation metrics: {metrics}")

        if model_name in ("xgboost", "random_forest", "gradient_boosting"):
            base_model = model.named_steps["clf"] if hasattr(model, "named_steps") else model
            log_shap_plot(base_model, pd.DataFrame(X_val, columns=feature_names), run_id)

        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, ARTIFACT_PATH,
                                     registered_model_name=REGISTERED_MODEL_NAME)
        else:
            mlflow.sklearn.log_model(model, ARTIFACT_PATH,
                                     registered_model_name=REGISTERED_MODEL_NAME)

        metrics_path = Path("training/last_run_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics["run_id"] = run_id
        metrics["model"] = model_name
        metrics_path.write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(metrics_path))

        print(f"\nRun complete. ROC-AUC: {metrics['roc_auc']}")
        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="churn-prediction")
    parser.add_argument("--model", default="xgboost", choices=list(MODELS.keys()))
    parser.add_argument("--data", default="data/processed/churn.csv")
    args = parser.parse_args()
    train(args.experiment, args.model, args.data)
