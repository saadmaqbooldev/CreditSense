import json

from scripts.train_xgboost_calibrated import train_calibrated


def test_train_calibrated_model_records_constraints_and_platt_scaling(tmp_path) -> None:
    model_path = tmp_path / "calibrated.pkl"
    metadata_path = tmp_path / "calibrated.json"

    metadata = train_calibrated(model_path=model_path, metadata_path=metadata_path)
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert model_path.exists()
    assert saved_metadata["model_type"] == "xgboost_platt_calibrated"
    assert saved_metadata["calibration"]["method"] == "sigmoid"
    assert saved_metadata["calibration"]["technique"] == "Platt scaling"
    assert saved_metadata["monotonic_constraints"]["debt_to_equity"] == 1
    assert saved_metadata["monotonic_constraints"]["collateral_coverage"] == -1
    assert 0.0 <= metadata["metrics"]["auc_roc"] <= 1.0
    assert 0.0 <= metadata["metrics"]["brier_score"] <= 1.0
