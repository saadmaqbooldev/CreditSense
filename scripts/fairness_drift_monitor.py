"""Run sector fairness and population-drift checks for the credit model."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "raw" / "creditsense_final_portfolio_v1.csv"
ARTIFACT_DIR = BASE_DIR / "app" / "ml_artifacts"
REPORT_PATH = ARTIFACT_DIR / "fairness_drift_report.json"
RANDOM_STATE = 42
CV_FOLDS = 5
PSI_ALERT_THRESHOLD = 0.20

sys.path.insert(0, str(BASE_DIR))
from app.ml_pipeline import FEATURE_COLUMNS, engineer_features  # noqa: E402
from scripts.evaluate_xgboost_model import build_estimator  # noqa: E402


def _frame(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(path)
    engineered = engineer_features(data)
    features = engineered[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target_column = next(
        (column for column in ("default_probability_12m", "default_probability") if column in data),
        None,
    )
    if target_column is None:
        raise ValueError("Dataset must contain default_probability or default_probability_12m")
    target = (pd.to_numeric(data[target_column], errors="raise") >= 0.5).astype(int)
    return data, pd.concat([features, target.rename("target")], axis=1)


def _group_metrics(data: pd.DataFrame, target: pd.Series, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    results: dict[str, dict[str, float]] = {}
    for sector in sorted(data["sector"].unique()):
        mask = data["sector"].to_numpy() == sector
        actual = target.to_numpy()[mask]
        predicted = predictions[mask]
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        results[sector] = {
            "sample_count": int(mask.sum()),
            "default_rate": float(actual.mean()),
            "selection_rate": float(predicted.mean()),
            "tpr_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        }
    metric_names = ("selection_rate", "tpr_recall", "fpr", "precision")
    gaps = {
        f"{metric}_max_gap": max(item[metric] for item in results.values())
        - min(item[metric] for item in results.values())
        for metric in metric_names
    }
    return {"by_sector": results, "max_gaps": gaps, "threshold": 0.5}


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference_values = pd.to_numeric(reference, errors="coerce").fillna(0.0)
    current_values = pd.to_numeric(current, errors="coerce").fillna(0.0)
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)
    reference_share = np.maximum(reference_counts / len(reference_values), 1e-6)
    current_share = np.maximum(current_counts / len(current_values), 1e-6)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def run_drift_check(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    numeric_features = [column for column in FEATURE_COLUMNS if column in reference]
    feature_psi = {
        column: _psi(reference[column], current[column]) for column in numeric_features
    }
    reference_sector = reference["sector"].value_counts(normalize=True)
    sector_psi = _psi(
        reference["sector"].map(reference_sector).fillna(0),
        current["sector"].map(reference_sector).fillna(0),
    )
    drift_values = {**feature_psi, "sector_distribution": sector_psi}
    return {
        "psi_by_feature": drift_values,
        "alert_threshold": PSI_ALERT_THRESHOLD,
        "alerts": sorted(
            feature for feature, value in drift_values.items() if value >= PSI_ALERT_THRESHOLD
        ),
        "sector_categories": sorted(set(reference["sector"]) | set(current["sector"])),
    }


def run_monitoring(
    reference_path: Path = DEFAULT_DATASET_PATH,
    current_path: Path = DEFAULT_DATASET_PATH,
    report_path: Path = REPORT_PATH,
) -> dict:
    reference_data, reference_frame = _frame(reference_path)
    current_data, _ = _frame(current_path)
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    probabilities = cross_val_predict(
        build_estimator(),
        reference_frame[FEATURE_COLUMNS],
        reference_frame["target"],
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    report = {
        "evaluation_version": "monitoring-v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "fairness": _group_metrics(reference_data, reference_frame["target"], probabilities),
        "drift": run_drift_check(reference_data, current_data),
        "reference_dataset": str(reference_path),
        "current_dataset": str(current_path),
    }
    report["all_drift_alerts_clear"] = not report["drift"]["alerts"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = run_monitoring()
    print(json.dumps({"fairness": report["fairness"]["max_gaps"], "drift_alerts": report["drift"]["alerts"]}, indent=2))
    print(f"Saved monitoring report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
