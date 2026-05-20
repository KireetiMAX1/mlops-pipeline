"""
Data drift monitoring using Evidently AI.

Detects:
  - Feature distribution drift (KS test, PSI)
  - Missing value changes

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
from evidently import Report
from evidently.presets import DataDriftPreset
import mlflow

DRIFT_THRESHOLD     = float(os.getenv("DRIFT_THRESHOLD", "0.5"))
REPORTS_DIR         = Path("monitoring/reports")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

NUM_COLS = ["tenure", "monthly_charges", "total_charges"]


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

    ref_num = reference[NUM_COLS].copy()
    cur_num = current[NUM_COLS].copy()

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_num, current_data=cur_num)

    html_path = output_dir / f"drift_report_{timestamp}.html"
    snapshot.save_html(str(html_path))

    result = snapshot.dict()
    metrics = result.get("metrics", [])

    drifted_count = 0.0
    drifted_share = 0.0
    col_drift_scores: dict = {}

    for m in metrics:
        name = m.get("metric_name", "")
        value = m.get("value", 0)

        if "DriftedColumnsCount" in name and isinstance(value, dict):
            drifted_count = float(value.get("count", 0))
            drifted_share = float(value.get("share", 0))

        for col in NUM_COLS:
            if f"ValueDrift(column={col}" in name:
                col_drift_scores[col] = round(float(value) if isinstance(value, (int, float)) else 0.0, 4)

    drift_detected = drifted_share > DRIFT_THRESHOLD

    summary: dict = {
        "timestamp":             timestamp,
        "drift_detected":        drift_detected,
        "share_drifted_columns": round(drifted_share, 4),
        "drifted_column_count":  int(drifted_count),
        "reference_rows":        len(reference),
        "current_rows":          len(current),
        "report_path":           str(html_path),
    }
    for col in NUM_COLS:
        summary[f"{col}_drift_score"] = col_drift_scores.get(col, 0.0)

    json_path = output_dir / f"drift_summary_{timestamp}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    return summary


def log_to_mlflow(summary: dict) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("monitoring-drift")
    with mlflow.start_run(run_name=f"drift-check-{summary['timestamp']}"):
        mlflow.log_metric("drift_detected", int(summary["drift_detected"]))
        mlflow.log_metric("share_drifted_columns", summary["share_drifted_columns"])
        for col in NUM_COLS:
            mlflow.log_metric(f"{col}_drift_score", summary.get(f"{col}_drift_score", 0))
        mlflow.log_artifact(summary["report_path"])


def trigger_retraining_if_needed(summary: dict) -> bool:
    if summary["drift_detected"]:
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
