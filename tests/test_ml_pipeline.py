from fastapi.testclient import TestClient

from app.main import app
from app.ml_pipeline import build_feature_vector, score_applicant


client = TestClient(app)


def test_build_feature_vector_has_expected_fields() -> None:
    payload = {
        "annual_revenue": 12000000,
        "debt_to_equity": 2.1,
        "revenue_growth_yoy": 0.14,
        "operating_margin": 0.18,
        "years_in_business": 7,
        "late_payment_count": 2,
        "collateral_coverage_ratio": 1.5,
    }

    features = build_feature_vector(payload)

    assert "debt_to_equity" in features
    assert "revenue_growth_yoy" in features
    assert "operating_margin" in features
    assert "years_in_business" in features
    assert "late_payment_count" in features
    assert "collateral_coverage_ratio" in features
    assert "debt_to_equity_sq" in features
    assert "leverage_risk" in features


def test_score_applicant_returns_probability_between_zero_and_one() -> None:
    payload = {
        "annual_revenue": 18000000,
        "debt_to_equity": 2.9,
        "revenue_growth_yoy": 0.08,
        "operating_margin": 0.12,
        "years_in_business": 4,
        "late_payment_count": 4,
        "collateral_coverage_ratio": 0.9,
    }

    score = score_applicant(payload)

    assert 0.0 <= score <= 1.0
    assert score > 0.3


def test_predict_credit_risk_endpoint_returns_model_response() -> None:
    payload = {
        "annual_revenue": 18000000,
        "debt_to_equity": 2.9,
        "revenue_growth_yoy": 0.08,
        "operating_margin": 0.12,
        "years_in_business": 4,
        "late_payment_count": 4,
        "collateral_coverage_ratio": 0.9,
    }

    response = client.post("/predict/credit-risk", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "default_probability" in body
    assert "risk_band" in body
    assert "model_version" in body
    assert 0.0 <= float(body["default_probability"]) <= 1.0
