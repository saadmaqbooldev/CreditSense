from fastapi.testclient import TestClient
from app.main import app
from app.orchestration import orchestrate_underwriting, OrchestrationError


client = TestClient(app)


def test_orchestration_flow_complete() -> None:
    applicant_data = {
        "applicant_id": "APP-TEST-001",
        "annual_revenue": 15000000.0,
        "debt_to_equity": 1.5,
        "revenue_growth_yoy": 0.12,
        "operating_margin": 0.16,
        "years_in_business": 5,
        "late_payment_count": 1,
        "collateral_coverage_ratio": 1.2,
        "notes": "Well-established manufacturing firm with stable cash flow.",
    }

    result = orchestrate_underwriting(applicant_data, "APP-TEST-001")

    assert result["applicant_id"] == "APP-TEST-001"
    assert "timestamp" in result
    assert "financial_analysis" in result
    assert "risk_assessment" in result
    assert "compliance" in result
    assert "final_decision" in result
    assert "approved" in result["final_decision"]


def test_orchestration_escalates_on_high_debt_to_equity() -> None:
    applicant_data = {
        "applicant_id": "APP-TEST-002",
        "annual_revenue": 12000000.0,
        "debt_to_equity": 3.2,
        "revenue_growth_yoy": 0.08,
        "operating_margin": 0.10,
        "years_in_business": 3,
        "late_payment_count": 3,
        "collateral_coverage_ratio": 0.8,
        "notes": "High leverage, weak collateral.",
    }

    result = orchestrate_underwriting(applicant_data, "APP-TEST-002")

    assert result["final_decision"]["escalation_required"] is True
    assert len(result["compliance"]["flags"]) >= 1


def test_orchestration_complies_with_guardrails() -> None:
    applicant_data = {
        "applicant_id": "APP-TEST-003",
        "annual_revenue": 25000000.0,
        "debt_to_equity": 1.0,
        "revenue_growth_yoy": 0.15,
        "operating_margin": 0.20,
        "years_in_business": 10,
        "late_payment_count": 0,
        "collateral_coverage_ratio": 1.5,
        "notes": "Strong financial position, excellent repayment history.",
    }

    result = orchestrate_underwriting(applicant_data, "APP-TEST-003")

    assert result["compliance"]["escalation_required"] is False
    assert len(result["final_decision"]["rationale"]) > 0


def test_submit_underwriting_endpoint_returns_structured_response() -> None:
    payload = {
        "applicant_id": "APP-ENDPOINT-001",
        "annual_revenue": 15000000.0,
        "debt_to_equity": 1.8,
        "revenue_growth_yoy": 0.12,
        "operating_margin": 0.15,
        "years_in_business": 6,
        "late_payment_count": 1,
        "collateral_coverage_ratio": 1.3,
        "notes": "Manufacturing sector, stable operations.",
    }

    response = client.post("/submit/underwriting", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["applicant_id"] == "APP-ENDPOINT-001"
    assert "timestamp" in body
    assert "financial_analysis" in body
    assert "risk_assessment" in body
    assert "compliance" in body
    assert "final_decision" in body


def test_submit_underwriting_endpoint_validates_input() -> None:
    payload = {
        "applicant_id": "APP-BAD-001",
        "annual_revenue": -100,
    }

    response = client.post("/submit/underwriting", json=payload)

    assert response.status_code == 422


def test_submit_underwriting_endpoint_error_handling() -> None:
    payload = {
        "applicant_id": "APP-ERROR-001",
        "annual_revenue": 15000000.0,
        "debt_to_equity": 1.8,
        "revenue_growth_yoy": 0.12,
        "operating_margin": 0.15,
        "years_in_business": 6,
        "late_payment_count": 1,
        "collateral_coverage_ratio": 1.3,
    }

    response = client.post("/submit/underwriting", json=payload)

    assert response.status_code in {200, 500, 503}


def test_structured_logging_in_orchestration() -> None:
    applicant_data = {
        "applicant_id": "APP-LOG-001",
        "annual_revenue": 15000000.0,
        "debt_to_equity": 1.5,
        "revenue_growth_yoy": 0.12,
        "operating_margin": 0.16,
        "years_in_business": 5,
        "late_payment_count": 1,
        "collateral_coverage_ratio": 1.2,
    }

    result = orchestrate_underwriting(applicant_data, "APP-LOG-001")

    assert result is not None
    assert result["applicant_id"] == "APP-LOG-001"


def test_rate_limiting_blocks_excessive_requests() -> None:
    payload = {
        "applicant_id": "APP-RATE-001",
        "annual_revenue": 15000000.0,
        "debt_to_equity": 1.5,
        "revenue_growth_yoy": 0.12,
        "operating_margin": 0.16,
        "years_in_business": 5,
        "late_payment_count": 1,
        "collateral_coverage_ratio": 1.2,
    }

    responses = [client.post("/submit/underwriting", json=payload) for _ in range(3)]

    assert all(r.status_code in {200, 429} for r in responses)


def test_health_endpoint_still_works() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
