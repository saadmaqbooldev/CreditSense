from __future__ import annotations

from pydantic import BaseModel, Field


class NormalizedFinancials(BaseModel):
    annual_revenue: float
    debt_to_equity: float
    revenue_growth_yoy: float
    operating_margin: float
    years_in_business: int
    late_payment_count: int
    collateral_coverage_ratio: float
    raw_notes: str = ""


class RiskAssessment(BaseModel):
    default_probability: float = Field(ge=0.0, le=1.0)
    risk_band: str
    narrative: str
    model_version: str


class RegulationSource(BaseModel):
    regulation_number: str
    title: str
    content: str


class ComplianceFlag(BaseModel):
    rule_id: str
    title: str
    severity: str
    rationale: str
    source: RegulationSource


class ComplianceResult(BaseModel):
    flags: list[ComplianceFlag]
    escalation_required: bool
    source_count: int
    query: str


class FinalDecision(BaseModel):
    approved: bool
    escalation_required: bool
    rationale: str
    risk_band: str
    compliance_flags_count: int


class UnderwritingResult(BaseModel):
    applicant_id: str
    timestamp: str
    financial_analysis: dict
    risk_assessment: RiskAssessment
    compliance: ComplianceResult
    final_decision: FinalDecision
