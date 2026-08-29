from __future__ import annotations

import math
from typing import Any


def build_feature_vector(record: dict[str, Any]) -> dict[str, float]:
    annual_revenue = float(record.get("annual_revenue", 0.0) or 0.0)
    debt_to_equity = float(record.get("debt_to_equity", 0.0) or 0.0)
    revenue_growth_yoy = float(record.get("revenue_growth_yoy", 0.0) or 0.0)
    operating_margin = float(record.get("operating_margin", 0.0) or 0.0)
    years_in_business = float(record.get("years_in_business", 0.0) or 0.0)
    late_payment_count = float(record.get("late_payment_count", 0.0) or 0.0)
    collateral_coverage_ratio = float(record.get("collateral_coverage_ratio", 0.0) or 0.0)

    leverage_risk = max(0.0, debt_to_equity - 1.2) / 2.0
    features = {
        "annual_revenue": annual_revenue,
        "debt_to_equity": debt_to_equity,
        "revenue_growth_yoy": revenue_growth_yoy,
        "operating_margin": operating_margin,
        "years_in_business": years_in_business,
        "late_payment_count": late_payment_count,
        "collateral_coverage_ratio": collateral_coverage_ratio,
        "debt_to_equity_sq": debt_to_equity**2,
        "leverage_risk": leverage_risk,
        "revenue_growth_z": max(0.0, revenue_growth_yoy),
        "margin_gap": max(0.0, 0.15 - operating_margin),
        "collateral_shortfall": max(0.0, 1.2 - collateral_coverage_ratio),
        "payment_risk": late_payment_count / max(1.0, years_in_business),
        "credit_history_signal": late_payment_count * 0.08,
    }
    return features


def score_applicant(record: dict[str, Any]) -> float:
    features = build_feature_vector(record)

    raw_score = (
        0.00000008 * features["annual_revenue"]
        + 0.18 * features["debt_to_equity"]
        + 0.22 * features["debt_to_equity_sq"]
        + 0.40 * features["leverage_risk"]
        + 0.80 * features["margin_gap"]
        + 0.55 * features["collateral_shortfall"]
        + 0.18 * features["payment_risk"]
        + 0.10 * features["credit_history_signal"]
        - 0.16 * features["revenue_growth_z"]
        - 0.05 * max(0.0, features["operating_margin"])
        - 0.02 * min(20.0, features["years_in_business"])
    )

    probability = 1 / (1 + math.exp(-(raw_score - 0.2)))
    return float(max(0.0, min(1.0, probability)))
