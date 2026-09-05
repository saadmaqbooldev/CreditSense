"""Train the Phase 3.2 baseline XGBoost classifier."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "raw" / "creditsense_final_portfolio_v1.csv"
ARTIFACT_DIR = BASE_DIR / "app" / "ml_artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgboost_model_baseline.pkl"
METADATA_PATH = ARTIFACT_DIR / "model_metadata_baseline.json"
RANDOM_STATE = 42
TEST_SIZE = 0.20

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
    target_values = pd.to_numeric(data[target_column], errors="raise")
    target = (target_values >= 0.5).astype(int)
    return features, target


def train_baseline(
    dataset_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> dict:
    features, target = load_training_frame(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = xgb.XGBClassifier(**PARAMETERS)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "auc_roc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "brier_score": brier_score_loss(y_test, probabilities),
        "specificity": true_negative / (true_negative + false_positive),
        "fpr": false_positive / (true_negative + false_positive),
        "threshold": 0.5,
        "confusion_matrix": [
            [int(true_negative), int(false_positive)],
            [int(false_negative), int(true_positive)],
        ],
    }
    metadata = {
        "model_type": "xgboost_baseline",
        "version": "baseline-v1",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "training_samples": len(x_train),
        "test_samples": len(x_test),
        "features": FEATURE_COLUMNS,
        "parameters": PARAMETERS,
        "metrics": metrics,
        "classification_report": classification_report(y_test, predictions),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    metadata = train_baseline()
    print(json.dumps(metadata["metrics"], indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
