from __future__ import annotations

from typing import Any
from datetime import datetime
import structlog

from app.agents import (
    ComplianceAgent,
    FinancialAnalystAgent,
    RiskScoringAgent,
    UnderwritingSupervisorAgent,
)


logger = structlog.get_logger(__name__)


class OrchestrationError(Exception):
    """Raised when orchestration encounters a recoverable error."""
    pass


def orchestrate_underwriting(
    applicant_data: dict[str, Any],
    applicant_id: str,
    retry_count: int = 0,
    max_retries: int = 2,
) -> dict[str, Any]:
    """
    Full orchestration flow: normalize → score → check compliance → decide.
    
    Error handling: retries once on transient failures, escalates on repeated errors.
    """
    try:
        logger.info("orchestration_start", applicant_id=applicant_id, retry_count=retry_count)

        analyst = FinancialAnalystAgent()
        normalized = analyst.normalize_financials(applicant_data)
        logger.info("financial_analysis_complete", applicant_id=applicant_id, years=normalized.years_in_business)
        normalized_data = normalized.model_dump()

        scorer = RiskScoringAgent()
        risk_result = scorer.assess(normalized_data)
        logger.info("risk_scoring_complete", applicant_id=applicant_id, risk_band=risk_result["risk_band"])

        compliance = ComplianceAgent()
        compliance_result = compliance.analyze(normalized_data)
        logger.info("compliance_check_complete", applicant_id=applicant_id, flags_count=len(compliance_result["flags"]))

        supervisor = UnderwritingSupervisorAgent()
        decision = supervisor.evaluate(
            default_probability=risk_result["default_probability"],
            risk_band=risk_result["risk_band"],
            compliance_flags=compliance_result["flags"],
        )
        logger.info("supervision_complete", applicant_id=applicant_id, approved=decision["approved"])

        return {
            "applicant_id": applicant_id,
            "timestamp": datetime.utcnow().isoformat(),
            "financial_analysis": {
                "normalized_revenue": normalized.annual_revenue,
                "normalized_debt_to_equity": normalized.debt_to_equity,
                "raw_notes": normalized.raw_notes,
            },
            "risk_assessment": {
                "default_probability": risk_result["default_probability"],
                "risk_band": risk_result["risk_band"],
                "narrative": risk_result["narrative"],
            },
            "compliance": {
                "flags": compliance_result["flags"],
                "escalation_required": compliance_result["escalation_required"],
            },
            "final_decision": {
                "approved": decision["approved"],
                "escalation_required": decision["escalation_required"],
                "rationale": decision["rationale"],
                "compliance_flags_count": decision["compliance_flags_count"],
            },
        }

    except Exception as e:
        logger.error("orchestration_error", applicant_id=applicant_id, error=str(e), retry_count=retry_count)

        if retry_count < max_retries:
            logger.info("orchestration_retry", applicant_id=applicant_id, attempt=retry_count + 1)
            return orchestrate_underwriting(applicant_data, applicant_id, retry_count + 1, max_retries)

        logger.error("orchestration_failed_final", applicant_id=applicant_id, attempts=retry_count + 1)
        raise OrchestrationError(f"Failed to process applicant {applicant_id} after {retry_count + 1} attempts: {str(e)}")
