"""Cross-validate and explain the constrained XGBoost model."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_validate

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "raw" / "creditsense_final_portfolio_v1.csv"
ARTIFACT_DIR = BASE_DIR / "app" / "ml_artifacts"
FEATURE_IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance_calibrated.csv"
SHAP_VALUES_PATH = ARTIFACT_DIR / "shap_values_calibrated.csv"
EVALUATION_PATH = ARTIFACT_DIR / "model_evaluation_calibrated.json"
RANDOM_STATE = 42
CV_FOLDS = 5

sys.path.insert(0, str(BASE_DIR))
from app.ml_pipeline import FEATURE_COLUMNS, engineer_features  # noqa: E402
from scripts.train_xgboost_calibrated import MONOTONIC_CONSTRAINTS, PARAMETERS  # noqa: E402


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


def build_estimator() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **PARAMETERS,
        monotone_constraints=MONOTONIC_CONSTRAINTS,
    )


def evaluate_model(
    dataset_path: Path = DATASET_PATH,
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict:
    features, target = load_training_frame(dataset_path)
    estimator = build_estimator()
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        estimator,
        features,
        target,
        cv=folds,
        scoring={"auc_roc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"},
        return_train_score=False,
        n_jobs=1,
    )

    estimator.fit(features, target)
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    sample = features.head(min(1000, len(features)))
    contribution_values = estimator.get_booster().predict(
        xgb.DMatrix(sample, feature_names=FEATURE_COLUMNS),
        pred_contribs=True,
    )
    contribution_frame = pd.DataFrame(
        contribution_values,
        columns=[*FEATURE_COLUMNS, "bias"],
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(artifact_dir / FEATURE_IMPORTANCE_PATH.name, index=False)
    contribution_frame.to_csv(artifact_dir / SHAP_VALUES_PATH.name, index=False)
    evaluation = {
        "model_version": "calibrated-v1",
        "evaluation_version": "evaluation-v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "samples": len(features),
        "cv_folds": CV_FOLDS,
        "cv_metrics": {
            "auc_roc_mean": float(np.mean(cv_results["test_auc_roc"])),
            "auc_roc_std": float(np.std(cv_results["test_auc_roc"])),
            "pr_auc_mean": float(np.mean(cv_results["test_pr_auc"])),
            "pr_auc_std": float(np.std(cv_results["test_pr_auc"])),
            "brier_score_mean": float(-np.mean(cv_results["test_brier"])),
            "brier_score_std": float(np.std(cv_results["test_brier"])),
        },
        "explainability": {
            "method": "XGBoost TreeSHAP contributions",
            "implementation": "Booster.predict(pred_contribs=True)",
            "sample_size": len(sample),
            "feature_importance_path": str(artifact_dir / FEATURE_IMPORTANCE_PATH.name),
            "shap_values_path": str(artifact_dir / SHAP_VALUES_PATH.name),
        },
        "parameters": PARAMETERS,
        "monotonic_constraints": MONOTONIC_CONSTRAINTS,
    }
    (artifact_dir / EVALUATION_PATH.name).write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return evaluation


def main() -> None:
    evaluation = evaluate_model()
    print(json.dumps(evaluation["cv_metrics"], indent=2))
    print(f"Saved evaluation: {EVALUATION_PATH}")


if __name__ == "__main__":
    main()
