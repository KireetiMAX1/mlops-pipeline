"""
Data pipeline: ingestion, validation, preprocessing, and versioning with DVC.

Run:
    python data/data_pipeline.py              # generate and validate data
    dvc add data/raw/churn.csv                # version with DVC
    dvc push                                  # push to remote storage
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass, field
from typing import List, Tuple


REQUIRED_COLUMNS: List[str] = [
    "tenure", "monthly_charges", "total_charges",
    "contract", "payment_method", "internet_service",
    "tech_support", "online_security", "churn",
]
NUMERIC_COLUMNS: List[str]     = ["tenure", "monthly_charges", "total_charges"]
CATEGORICAL_COLUMNS: List[str] = [
    "contract", "payment_method", "internet_service", "tech_support", "online_security"
]
TARGET: str  = "churn"
MIN_ROWS: int = 100


@dataclass
class ValidationReport:
    passed: bool = True
    issues: List[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.issues.append(msg)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "issues": self.issues}


def validate_schema(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.fail(f"Missing columns: {missing}")

    if len(df) < MIN_ROWS:
        report.fail(f"Too few rows: {len(df)} < {MIN_ROWS}")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            null_pct = df[col].isnull().mean()
            if null_pct > 0.05:
                report.fail(f"Column '{col}' has {null_pct:.1%} nulls (>5%)")

    if TARGET in df.columns:
        target_rate = df[TARGET].mean()
        if target_rate < 0.05 or target_rate > 0.95:
            report.fail(f"Extreme class imbalance: positive rate = {target_rate:.2%}")

    return report


def generate_sample_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure       = rng.integers(1, 72, n)
    monthly      = rng.uniform(20, 120, n).round(2)
    total        = (tenure * monthly * rng.uniform(0.9, 1.1, n)).round(2)
    contract     = rng.choice(["Month-to-month", "One year", "Two year"],
                              n, p=[0.55, 0.25, 0.20])
    payment      = rng.choice(["Electronic check", "Mailed check",
                               "Bank transfer", "Credit card"], n)
    internet     = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.44, 0.34, 0.22])
    tech_support = rng.choice(["Yes", "No", "No internet"], n)
    online_sec   = rng.choice(["Yes", "No", "No internet"], n)

    churn_prob = (
        0.05
        + 0.25 * (contract == "Month-to-month")
        + 0.10 * (internet == "Fiber optic")
        - 0.15 * (tenure > 24)
        + 0.08 * (payment == "Electronic check")
        - 0.05 * (tech_support == "Yes")
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = rng.binomial(1, churn_prob, n)

    return pd.DataFrame({
        "tenure": tenure, "monthly_charges": monthly, "total_charges": total,
        "contract": contract, "payment_method": payment,
        "internet_service": internet, "tech_support": tech_support,
        "online_security": online_sec, "churn": churn,
    })


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()

    le = LabelEncoder()
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    df["charges_per_tenure"] = df["monthly_charges"] / (df["tenure"] + 1)
    df["high_value_customer"] = (df["monthly_charges"] > 70).astype(int)

    feature_cols = [c for c in df.columns if c != TARGET]
    return df[feature_cols], feature_cols


def load_and_preprocess(
    data_path: str, test_size: float = 0.2, seed: int = 42
) -> Tuple[object, object, object, object, List[str]]:
    path = Path(data_path)
    if not path.exists():
        print(f"  Data not found at {path} — generating synthetic dataset...")
        path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_sample_data()
        df.to_csv(path, index=False)
        print(f"  Saved {len(df)} rows to {path}")
    else:
        df = pd.read_csv(path)
        print(f"  Loaded {len(df)} rows from {path}")

    report = validate_schema(df)
    report_path = path.parent / "validation_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    if not report.passed:
        raise ValueError(f"Data validation failed: {report.issues}")
    print("  Data validation passed")

    X, feature_names = preprocess(df)
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return X_train.values, X_val.values, y_train.values, y_val.values, feature_names


if __name__ == "__main__":
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df = generate_sample_data(n=5000)
    df.to_csv("data/raw/churn.csv", index=False)
    df.to_csv("data/processed/churn.csv", index=False)
    report = validate_schema(df)
    print(f"Generated dataset: {len(df)} rows | Churn rate: {df['churn'].mean():.1%}")
    print(f"Validation: {'PASSED' if report.passed else 'FAILED'}")
    if not report.passed:
        for issue in report.issues:
            print(f"  - {issue}")
