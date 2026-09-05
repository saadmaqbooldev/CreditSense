import json

from scripts.train_xgboost_baseline import train_baseline


def test_train_baseline_writes_model_and_metrics(tmp_path) -> None:
    model_path = tmp_path / "baseline.pkl"
    metadata_path = tmp_path / "baseline.json"

    metadata = train_baseline(
        model_path=model_path,
        metadata_path=metadata_path,
    )
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert model_path.exists()
    assert saved_metadata["model_type"] == "xgboost_baseline"
    assert saved_metadata["training_samples"] == 12_000
    assert saved_metadata["test_samples"] == 3_000
    assert 0.0 <= metadata["metrics"]["auc_roc"] <= 1.0
    assert 0.0 <= metadata["metrics"]["pr_auc"] <= 1.0
