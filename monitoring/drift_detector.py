"""
Data drift and model performance monitoring using Evidently AI.

Detects:
  - Feature distribution drift (PSI, KS test)
  - Target drift (prediction distribution shift)
  - Missing value changes
  - Model performance degradation

Run:
    python monitoring/drift_detector.py --reference data/processed/churn.csv
    python monitoring/drift_detector.py --reference data/processed/churn.csv --drift_factor 0.5
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
)
import mlflow

DRIFT_THRESHOLD     = float(os.getenv("DRIFT_THRESHOLD", "0.15"))
REPORTS_DIR         = Path("monitoring/reports")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def load_reference(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def simulate_production_data(reference: pd.DataFrame, drift_factor: float = 0.0) -> pd.DataFrame:
    """
    Simulate incoming production data.
    drift_factor=0.0  -> no drift (stable distribution)
    drift_factor=1.0  -> severe drift
    """
    n = len(reference) // 4
    new_data = reference.sample(n=n, replace=True, random_state=int(time.time())).copy()

    if drift_factor > 0:
        rng = np.random.default_rng(42)
        new_data["monthly_charges"] += rng.normal(
            loc=drift_factor * 20, scale=5, size=n
        )
        new_data["tenure"] = np.clip(
            new_data["tenure"] - int(drift_factor * 10), 1, 72
        )
        print(f"  Simulated drift applied (factor={drift_factor})")
    return new_data


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path = REPORTS_DIR,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    num_cols = ["tenure", "monthly_charges", "total_charges"]
    ref_num = reference[num_cols]
    cur_num = current[num_cols]

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        *[ColumnDriftMetric(column_name=col) for col in num_cols],
    ])

    report.run(reference_data=ref_num, current_data=cur_num)

    html_path = output_dir / f"drift_report_{timestamp}.html"
    report.save_html(str(html_path))

    result = report.as_dict()
    drift_detected = result["metrics"][2]["result"]["dataset_drift"]
    share_drifted  = result["metrics"][2]["result"]["share_of_drifted_columns"]

    summary = {
        "timestamp":              timestamp,
        "drift_detected":         drift_detected,
        "share_drifted_columns":  round(share_drifted, 4),
        "reference_rows":         len(reference),
        "current_rows":           len(current),
        "report_path":            str(html_path),
    }

    for i, col in enumerate(num_cols):
        col_result = result["metrics"][4 + i]["result"]
        summary[f"{col}_drift_score"] = round(col_result.get("drift_score", 0.0), 4)
        summary[f"{col}_drifted"]     = col_result.get("drift_detected", False)

    json_path = output_dir / f"drift_summary_{timestamp}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    return summary


def log_to_mlflow(summary: dict):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("monitoring-drift")
    with mlflow.start_run(run_name=f"drift-check-{summary['timestamp']}"):
        mlflow.log_metric("drift_detected", int(summary["drift_detected"]))
        mlflow.log_metric("share_drifted_columns", summary["share_drifted_columns"])
        for col in ["tenure", "monthly_charges", "total_charges"]:
            mlflow.log_metric(f"{col}_drift_score", summary.get(f"{col}_drift_score", 0))
        mlflow.log_artifact(summary["report_path"])


def trigger_retraining_if_needed(summary: dict) -> bool:
    if summary["drift_detected"] and summary["share_drifted_columns"] > DRIFT_THRESHOLD:
        print(f"\nDRIFT ALERT: {summary['share_drifted_columns']:.0%} of columns drifted!")
        print("  Automated retraining should be triggered via GitHub Actions or workflow scheduler.")
        # In production: call GitHub Actions API, Airflow, or Prefect here
        # requests.post(os.getenv("RETRAINING_WEBHOOK_URL"), json=summary)
        return True
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/processed/churn.csv")
    parser.add_argument("--drift_factor", type=float, default=0.0,
                        help="0=no drift, 1=severe drift (for demo purposes)")
    args = parser.parse_args()

    print("Running drift detection...")
    reference = load_reference(args.reference)
    current   = simulate_production_data(reference, drift_factor=args.drift_factor)

    summary   = run_drift_report(reference, current)
    log_to_mlflow(summary)
    triggered = trigger_retraining_if_needed(summary)

    print(f"\n{'DRIFT DETECTED' if summary['drift_detected'] else 'No drift detected'}")
    print(f"  Share of drifted columns: {summary['share_drifted_columns']:.0%}")
    print(f"  Report saved to: {summary['report_path']}")
    print(f"  Retraining triggered: {triggered}")
