import json

from scripts.evaluate_xgboost_model import evaluate_model


def test_evaluate_model_writes_cv_and_explainability_artifacts(tmp_path) -> None:
    evaluation = evaluate_model(artifact_dir=tmp_path)

    feature_importance_path = tmp_path / "feature_importance_calibrated.csv"
    shap_values_path = tmp_path / "shap_values_calibrated.csv"
    evaluation_path = tmp_path / "model_evaluation_calibrated.json"

    assert feature_importance_path.exists()
    assert shap_values_path.exists()
    assert evaluation_path.exists()
    assert evaluation["cv_folds"] == 5
    assert evaluation["explainability"]["method"] == "XGBoost TreeSHAP contributions"
    saved_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert 0.0 <= saved_evaluation["cv_metrics"]["auc_roc_mean"] <= 1.0
    assert saved_evaluation["explainability"]["sample_size"] == 1000
