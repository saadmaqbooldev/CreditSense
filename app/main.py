from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import structlog

from app.config import get_settings
from app.ml_pipeline import score_applicant
from app.orchestration import orchestrate_underwriting, OrchestrationError
from app.middleware import RateLimitMiddleware, StructuredLoggingMiddleware

settings = get_settings()
logger = structlog.get_logger(__name__)

app = FastAPI(title=settings.app_name, version="0.1.0")

app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
app.add_middleware(StructuredLoggingMiddleware)


class CreditRiskRequest(BaseModel):
    annual_revenue: float = Field(..., gt=0)
    debt_to_equity: float = Field(..., ge=0)
    revenue_growth_yoy: float = Field(...)
    operating_margin: float = Field(...)
    years_in_business: int = Field(..., gt=0)
    late_payment_count: int = Field(..., ge=0)
    collateral_coverage_ratio: float = Field(..., ge=0)


class CreditRiskResponse(BaseModel):
    default_probability: float
    risk_band: str
    model_version: str


class UnderwritingRequest(BaseModel):
    applicant_id: str
    annual_revenue: float = Field(..., gt=0)
    debt_to_equity: float = Field(..., ge=0)
    revenue_growth_yoy: float = Field(...)
    operating_margin: float = Field(...)
    years_in_business: int = Field(..., gt=0)
    late_payment_count: int = Field(..., ge=0)
    collateral_coverage_ratio: float = Field(..., ge=0)
    notes: str = ""


class UnderwritingResponse(BaseModel):
    applicant_id: str
    timestamp: str
    financial_analysis: dict
    risk_assessment: dict
    compliance: dict
    final_decision: dict


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name, "environment": settings.environment}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "CreditSense API is running."}


@app.post("/predict/credit-risk", response_model=CreditRiskResponse)
def predict_credit_risk(payload: CreditRiskRequest) -> CreditRiskResponse:
    probability = score_applicant(payload.model_dump())

    if probability >= 0.6:
        risk_band = "high"
    elif probability >= 0.35:
        risk_band = "moderate"
    else:
        risk_band = "low"

    return CreditRiskResponse(
        default_probability=round(probability, 6),
        risk_band=risk_band,
        model_version="baseline-v1",
    )


@app.post("/submit/underwriting", response_model=UnderwritingResponse)
def submit_underwriting(payload: UnderwritingRequest) -> UnderwritingResponse:
    """
    Full orchestration endpoint: normalizes financials → scores risk → checks compliance → decides.
    Implements hard guardrails and structured audit logging.
    """
    try:
        logger.info("underwriting_request_received", applicant_id=payload.applicant_id)
        
        result = orchestrate_underwriting(payload.model_dump(), payload.applicant_id)
        
        logger.info(
            "underwriting_decision_final",
            applicant_id=payload.applicant_id,
            approved=result["final_decision"]["approved"],
        )
        
        return UnderwritingResponse(**result)
    
    except OrchestrationError as e:
        logger.error("underwriting_orchestration_failed", applicant_id=payload.applicant_id, error=str(e))
        raise HTTPException(status_code=503, detail=f"Underwriting processing failed: {str(e)}")
    
    except Exception as e:
        logger.error("underwriting_unexpected_error", applicant_id=payload.applicant_id, error=str(e))
        raise HTTPException(status_code=500, detail="Unexpected error during underwriting processing")
