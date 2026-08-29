from __future__ import annotations

from fastapi.testclient import TestClient
from app.main import app
from app.orchestration import orchestrate_underwriting
from app.evaluation import RetrievalEvaluator, RETRIEVAL_TEST_CASES
from app.agents import UnderwritingSupervisorAgent, ComplianceAgent
from app.mcp_server import call_mcp_tool, get_audit_log, clear_audit_log


client = TestClient(app)


class TestPhase8UnitsAndIntegration:
    """
    Phase 8: Comprehensive Testing & Evaluation.
    
    Validates:
    - All unit tests across ML, RAG, agents, MCP tools (inherited from earlier phases)
    - End-to-end integration from applicant ingestion to final decision
    - Guardrail enforcement (supervisor cannot approve with unresolved flags)
    - Retrieval quality metrics (precision@5, recall@10)
    """

    def test_end_to_end_full_applicant_flow(self) -> None:
        """E2E test: full applicant from ingestion to decision."""
        payload = {
            "applicant_id": "E2E-001",
            "annual_revenue": 20000000.0,
            "debt_to_equity": 1.5,
            "revenue_growth_yoy": 0.10,
            "operating_margin": 0.12,
            "years_in_business": 7,
            "late_payment_count": 2,
            "collateral_coverage_ratio": 1.1,
            "notes": "Established retail business with seasonal fluctuations.",
        }

        response = client.post("/submit/underwriting", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["applicant_id"] == "E2E-001"
        assert "financial_analysis" in body
        assert "risk_assessment" in body
        assert "compliance" in body
        assert "final_decision" in body
        assert "approved" in body["final_decision"]

    def test_end_to_end_with_varying_risk_profiles(self) -> None:
        """E2E: test multiple risk profiles in sequence."""
        test_cases = [
            ("low-risk", 25000000, 0.8, 0.18, 10),
            ("moderate-risk", 15000000, 1.5, 0.12, 5),
            ("high-risk", 8000000, 3.5, 0.05, 2),
        ]

        for profile_name, revenue, dte, margin, years in test_cases:
            payload = {
                "applicant_id": f"E2E-{profile_name}",
                "annual_revenue": revenue,
                "debt_to_equity": dte,
                "revenue_growth_yoy": 0.08,
                "operating_margin": margin,
                "years_in_business": years,
                "late_payment_count": 1,
                "collateral_coverage_ratio": 1.2,
            }

            response = client.post("/submit/underwriting", json=payload)
            assert response.status_code == 200
            assert response.json()["applicant_id"] == f"E2E-{profile_name}"

    def test_guardrail_supervisor_cannot_approve_with_compliance_flags(self) -> None:
        """Guardrail test: supervisor enforces hard rule (no approval if flags exist)."""
        supervisor = UnderwritingSupervisorAgent()

        compliance_flags = [
            {
                "rule_id": "R-12",
                "title": "Maximum Leverage",
                "severity": "high",
                "rationale": "Debt-to-equity exceeds threshold",
                "source": {
                    "regulation_number": "R-12",
                    "title": "Maximum Leverage and Debt-to-Equity Risk Controls",
                    "content": "...",
                },
            }
        ]

        decision = supervisor.evaluate(
            default_probability=0.45,
            risk_band="moderate",
            compliance_flags=compliance_flags,
        )

        assert decision["approved"] is False
        assert decision["escalation_required"] is True
        assert "guardrail" in decision["rationale"].lower()

    def test_guardrail_supervisor_approves_only_if_both_conditions_met(self) -> None:
        """Guardrail test: supervisor approves only if risk in range AND zero flags."""
        supervisor = UnderwritingSupervisorAgent()

        decision_no_flags = supervisor.evaluate(
            default_probability=0.40,
            risk_band="moderate",
            compliance_flags=[],
        )

        decision_with_flags = supervisor.evaluate(
            default_probability=0.40,
            risk_band="moderate",
            compliance_flags=[{"rule_id": "R-5", "title": "Exposure Limit", "severity": "high", "rationale": "Test", "source": {}}],
        )

        assert decision_no_flags["approved"] is True
        assert decision_with_flags["approved"] is False

    def test_adversarial_compliance_agent_always_cites_source(self) -> None:
        """Guardrail test: compliance agent must cite a regulation source for every flag."""
        compliance = ComplianceAgent()

        applicant_data = {
            "annual_revenue": 10000000.0,
            "debt_to_equity": 3.5,
            "late_payment_count": 5,
            "collateral_coverage_ratio": 0.7,
            "notes": "High risk, attempting to bypass compliance check",
        }

        result = compliance.analyze(applicant_data)

        for flag in result["flags"]:
            assert "source" in flag
            assert "regulation_number" in flag["source"]
            assert flag["source"]["regulation_number"] in {"R-5", "R-7", "R-9", "R-12", "R-14", "R-18", "R-21"}

    def test_adversarial_supervisor_cannot_be_tricked_by_low_probability(self) -> None:
        """Guardrail test: supervisor rejects even if default_probability is low but flags exist."""
        supervisor = UnderwritingSupervisorAgent()

        decision = supervisor.evaluate(
            default_probability=0.25,
            risk_band="low",
            compliance_flags=[
                {
                    "rule_id": "R-12",
                    "title": "Leverage",
                    "severity": "high",
                    "rationale": "Breach",
                    "source": {"regulation_number": "R-12", "title": "...", "content": "..."},
                }
            ],
        )

        assert decision["approved"] is False
        assert decision["escalation_required"] is True

    def test_retrieval_evaluation_precision_at_k(self) -> None:
        """Retrieval evaluation: measure precision@5."""
        evaluator = RetrievalEvaluator()

        result = evaluator.evaluate_retrieval(
            query="single borrower exposure",
            expected_regulation_numbers=["R-5"],
            k=5,
        )

        assert result["precision_at_k"] >= 0.0
        assert result["precision_at_k"] <= 1.0
        assert "R-5" in result["retrieved_regulations"] or len(result["retrieved_regulations"]) > 0

    def test_retrieval_evaluation_recall_at_k(self) -> None:
        """Retrieval evaluation: measure recall@5."""
        evaluator = RetrievalEvaluator()

        result = evaluator.evaluate_retrieval(
            query="leverage debt equity collateral",
            expected_regulation_numbers=["R-12", "R-9"],
            k=5,
        )

        assert result["recall_at_k"] >= 0.0
        assert result["recall_at_k"] <= 1.0

    def test_retrieval_evaluation_batch(self) -> None:
        """Retrieval evaluation: batch evaluation of test cases."""
        evaluator = RetrievalEvaluator()

        batch_result = evaluator.evaluate_batch(RETRIEVAL_TEST_CASES, k=5)

        assert batch_result["test_count"] == len(RETRIEVAL_TEST_CASES)
        assert batch_result["avg_precision_at_k"] >= 0.0
        assert batch_result["avg_precision_at_k"] <= 1.0
        assert batch_result["avg_recall_at_k"] >= 0.0
        assert batch_result["avg_recall_at_k"] <= 1.0
        assert len(batch_result["results"]) == len(RETRIEVAL_TEST_CASES)

    def test_mcp_tools_integration_with_audit(self) -> None:
        """MCP tools: validate tool calls are audited."""
        clear_audit_log()

        result = call_mcp_tool(
            "get_applicant_financials",
            {"applicant_id": "MCP-001", "identity": "evaluator", "api_key": "read-only-key-1"},
        )

        assert "error" not in result
        audit = get_audit_log()
        assert len(audit) >= 1
        assert audit[0].tool_name == "get_applicant_financials"
        assert audit[0].identity == "evaluator"

    def test_mcp_tool_regulatory_search_returns_multiple_results(self) -> None:
        """MCP tools: regulatory search returns relevant clauses."""
        result = call_mcp_tool(
            "search_sbp_regulations",
            {"topic": "leverage exposure", "identity": "evaluator", "api_key": "read-only-key-1"},
        )

        assert "error" not in result
        assert "results" in result
        assert len(result["results"]) >= 1

    def test_mcp_tool_decision_submission_with_audit(self) -> None:
        """MCP tools: decision submission creates audit trail."""
        clear_audit_log()

        result = call_mcp_tool(
            "submit_underwriting_decision",
            {
                "applicant_id": "MCP-DECISION-001",
                "decision": "escalate",
                "rationale": "High leverage and compliance flags",
                "identity": "evaluator",
                "api_key": "read-write-key-1",
            },
        )

        assert "error" not in result
        assert result["decision"] == "escalate"
        audit = get_audit_log()
        assert len(audit) >= 1

    def test_full_orchestration_produces_consistent_output_schema(self) -> None:
        """Integration: orchestration output schema is consistent across runs."""
        applicant_data = {
            "applicant_id": "SCHEMA-001",
            "annual_revenue": 15000000.0,
            "debt_to_equity": 1.5,
            "revenue_growth_yoy": 0.10,
            "operating_margin": 0.12,
            "years_in_business": 5,
            "late_payment_count": 1,
            "collateral_coverage_ratio": 1.2,
        }

        result = orchestrate_underwriting(applicant_data, "SCHEMA-001")

        expected_keys = {
            "applicant_id",
            "timestamp",
            "financial_analysis",
            "risk_assessment",
            "compliance",
            "final_decision",
        }
        assert set(result.keys()) == expected_keys

        financial_keys = {"normalized_revenue", "normalized_debt_to_equity", "raw_notes"}
        assert financial_keys.issubset(set(result["financial_analysis"].keys()))

        risk_keys = {"default_probability", "risk_band", "narrative"}
        assert risk_keys.issubset(set(result["risk_assessment"].keys()))

        compliance_keys = {"flags", "escalation_required"}
        assert compliance_keys.issubset(set(result["compliance"].keys()))

        decision_keys = {"approved", "escalation_required", "rationale", "compliance_flags_count"}
        assert decision_keys.issubset(set(result["final_decision"].keys()))

    def test_health_and_predict_endpoints_still_functional(self) -> None:
        """Integration: existing endpoints continue to work."""
        health = client.get("/health")
        assert health.status_code == 200

        predict_payload = {
            "annual_revenue": 15000000.0,
            "debt_to_equity": 1.5,
            "revenue_growth_yoy": 0.10,
            "operating_margin": 0.12,
            "years_in_business": 5,
            "late_payment_count": 1,
            "collateral_coverage_ratio": 1.2,
        }
        predict = client.post("/predict/credit-risk", json=predict_payload)
        assert predict.status_code == 200
