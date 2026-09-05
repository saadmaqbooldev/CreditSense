"""Train the constrained and Platt-calibrated XGBoost model for Phase 3."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "raw" / "creditsense_final_portfolio_v1.csv"
ARTIFACT_DIR = BASE_DIR / "app" / "ml_artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgboost_model_calibrated.pkl"
METADATA_PATH = ARTIFACT_DIR / "model_metadata_calibrated.json"
RANDOM_STATE = 42

sys.path.insert(0, str(BASE_DIR))
from app.ml_pipeline import FEATURE_COLUMNS, engineer_features  # noqa: E402


PARAMETERS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Positive values increase risk; negative values decrease risk.
MONOTONIC_CONSTRAINTS = {
    "leverage": 1,
    "debt_to_equity": 1,
    "revenue_growth_yoy": -1,
    "years_in_business": -1,
    "late_payment_count": 1,
    "collateral_coverage": -1,
    "debt_service_ratio": 1,
    "liquidity_ratio": -1,
    "bank_statement_volatility": 1,
    "sector_risk_score": 1,
    "current_ratio": -1,
    "leverage_x_growth": 1,
    "debt_x_collateral": 1,
    "payment_history": 1,
}


def load_training_frame(dataset_path: Path = DATASET_PATH) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(dataset_path)
    engineered = engineer_features(data)
    features = engineered[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target_column = next(
        (column for column in ("default_probability_12m", "default_probability") if column in data),
        None,
    )
    if target_column is None:
        raise ValueError("Dataset must contain default_probability or default_probability_12m")
    target = (pd.to_numeric(data[target_column], errors="raise") >= 0.5).astype(int)
    return features, target


def train_calibrated(
    dataset_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> dict:
    features, target = load_training_frame(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    base_model = xgb.XGBClassifier(
        **PARAMETERS,
        monotone_constraints=MONOTONIC_CONSTRAINTS,
    )
    model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=3,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "auc_roc": roc_auc_score(y_test, probabilities),
        "brier_score": brier_score_loss(y_test, probabilities),
    }
    metadata = {
        "model_type": "xgboost_platt_calibrated",
        "version": "calibrated-v1",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "training_samples": len(x_train),
        "test_samples": len(x_test),
        "features": FEATURE_COLUMNS,
        "parameters": PARAMETERS,
        "monotonic_constraints": MONOTONIC_CONSTRAINTS,
        "calibration": {
            "method": "sigmoid",
            "technique": "Platt scaling",
            "cv_folds": 3,
        },
        "metrics": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    metadata = train_calibrated()
    print(json.dumps(metadata["metrics"], indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
