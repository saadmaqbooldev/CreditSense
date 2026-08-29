from app.agents import (
    ComplianceAgent,
    FinancialAnalystAgent,
    UnderwritingSupervisorAgent,
)


def test_financial_analyst_agent_normalizes_values_and_notes() -> None:
    raw = {
        "annual_revenue": "12000000",
        "debt_to_equity": "2.4",
        "revenue_growth_yoy": "0.12",
        "operating_margin": "0.18",
        "years_in_business": "5",
        "late_payment_count": "3",
        "collateral_coverage_ratio": "1.1",
        "notes": "client ka cash flow seasonal hai, Eid se pehle spike hota hai",
    }

    normalized = FinancialAnalystAgent().normalize_financials(raw)

    assert normalized.annual_revenue == 12000000.0
    assert normalized.debt_to_equity == 2.4
    assert normalized.years_in_business == 5
    assert "cash flow" in normalized.raw_notes.lower()
    assert "seasonal" in normalized.raw_notes.lower()


def test_compliance_agent_uses_retrieved_sources() -> None:
    agent = ComplianceAgent()

    result = agent.analyze(
        {
            "loan_amount": 2500000,
            "debt_to_equity": 2.8,
            "late_payment_count": 4,
            "collateral_coverage_ratio": 0.9,
            "notes": "high leverage and delayed repayments",
        }
    )

    assert result["escalation_required"] is True
    assert len(result["flags"]) >= 1
    assert result["flags"][0]["source"]["regulation_number"] in {"R-12", "R-14", "R-9"}


def test_underwriting_supervisor_rejects_noncompliant_decision() -> None:
    supervisor = UnderwritingSupervisorAgent()

    decision = supervisor.evaluate(
        default_probability=0.42,
        risk_band="moderate",
        compliance_flags=[
            {
                "rule_id": "R-12",
                "title": "Maximum Leverage and Debt-to-Equity Risk Controls",
                "severity": "high",
                "rationale": "Debt-to-equity exceeds policy thresholds.",
                "source": {
                    "regulation_number": "R-12",
                    "title": "Maximum Leverage and Debt-to-Equity Risk Controls",
                    "content": "Debt-to-equity ratios above policy thresholds may require additional approval.",
                },
            }
        ],
    )

    assert decision["approved"] is False
    assert decision["escalation_required"] is True
    assert "guardrail" in decision["rationale"].lower()
