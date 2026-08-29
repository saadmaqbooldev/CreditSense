from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.ml_pipeline import score_applicant
from app.rag_pipeline import build_regulation_chunks, retrieve_relevant_chunks


@dataclass
class NormalizedFinancials:
    annual_revenue: float
    debt_to_equity: float
    revenue_growth_yoy: float
    operating_margin: float
    years_in_business: int
    late_payment_count: int
    collateral_coverage_ratio: float
    raw_notes: str = ""


class FinancialAnalystAgent:
    """Normalizes raw applicant inputs and keeps audit-friendly notes."""

    def normalize_financials(self, raw: dict[str, Any]) -> NormalizedFinancials:
        notes = str(raw.get("notes") or "")
        return NormalizedFinancials(
            annual_revenue=float(raw.get("annual_revenue", 0.0) or 0.0),
            debt_to_equity=float(raw.get("debt_to_equity", 0.0) or 0.0),
            revenue_growth_yoy=float(raw.get("revenue_growth_yoy", 0.0) or 0.0),
            operating_margin=float(raw.get("operating_margin", 0.0) or 0.0),
            years_in_business=int(float(raw.get("years_in_business", 0) or 0)),
            late_payment_count=int(float(raw.get("late_payment_count", 0) or 0)),
            collateral_coverage_ratio=float(raw.get("collateral_coverage_ratio", 0.0) or 0.0),
            raw_notes=notes,
        )


class RiskScoringAgent:
    """Scores the normalized applicant and translates the result into a narrative."""

    def assess(self, applicant: dict[str, Any]) -> dict[str, Any]:
        score = score_applicant(applicant)
        if score >= 0.6:
            risk_band = "high"
        elif score >= 0.35:
            risk_band = "moderate"
        else:
            risk_band = "low"

        return {
            "default_probability": round(score, 6),
            "risk_band": risk_band,
            "narrative": (
                f"This applicant sits in the {risk_band} risk band with an estimated default "
                f"probability of {score:.2%}."
            ),
            "model_version": "baseline-v1",
        }


class ComplianceAgent:
    """Checks proposed structure against relevant SBP clauses and requires cited source chunks."""

    def __init__(self) -> None:
        self.regulation_chunks = build_regulation_chunks()

    def analyze(self, applicant: dict[str, Any]) -> dict[str, Any]:
        query = " ".join(
            [
                str(applicant.get("notes") or ""),
                f"debt_to_equity {applicant.get('debt_to_equity', '')}",
                f"late_payment_count {applicant.get('late_payment_count', '')}",
                f"collateral_coverage_ratio {applicant.get('collateral_coverage_ratio', '')}",
            ]
        )
        matches = retrieve_relevant_chunks(query, self.regulation_chunks)

        flags: list[dict[str, Any]] = []
        if float(applicant.get("debt_to_equity", 0.0) or 0.0) > 2.0:
            source = matches[0] if matches else self.regulation_chunks[0]
            flags.append(
                {
                    "rule_id": source["regulation_number"],
                    "title": source["title"],
                    "severity": "high",
                    "rationale": "Debt-to-equity exceeds the acceptable underwriting threshold and requires escalation.",
                    "source": source,
                }
            )
        if float(applicant.get("late_payment_count", 0.0) or 0.0) >= 3:
            source = next((m for m in matches if m['regulation_number'] == 'R-14'), self.regulation_chunks[0])
            flags.append(
                {
                    "rule_id": source["regulation_number"],
                    "title": source["title"],
                    "severity": "medium",
                    "rationale": "Late payment history indicates elevated delinquency risk and requires review.",
                    "source": source,
                }
            )
        if float(applicant.get("collateral_coverage_ratio", 0.0) or 0.0) < 1.0:
            source = next((m for m in matches if m['regulation_number'] == 'R-9'), self.regulation_chunks[0])
            flags.append(
                {
                    "rule_id": source["regulation_number"],
                    "title": source["title"],
                    "severity": "medium",
                    "rationale": "Collateral coverage is below the minimum prudent threshold.",
                    "source": source,
                }
            )

        escalation_required = len(flags) > 0
        return {
            "flags": flags,
            "escalation_required": escalation_required,
            "source_count": len(matches),
            "query": query,
        }


class UnderwritingSupervisorAgent:
    """Hard approval guardrail: approve only if risk is in range and no unresolved compliance flags remain."""

    def evaluate(
        self,
        default_probability: float,
        risk_band: str,
        compliance_flags: list[dict[str, Any]],
    ) -> dict[str, Any]:
        in_range = risk_band in {"low", "moderate"}
        compliant = len(compliance_flags) == 0

        approved = bool(in_range and compliant and default_probability < 0.6)
        rationale = (
            "Guardrail enforced: approval requires a non-high risk profile and zero unresolved compliance flags. "
            "Any active rule breach blocks approval."
        ) if not approved else "All underwriting checks passed under the strict guardrail."

        return {
            "approved": approved,
            "escalation_required": not approved,
            "rationale": rationale,
            "risk_band": risk_band,
            "compliance_flags_count": len(compliance_flags),
        }
