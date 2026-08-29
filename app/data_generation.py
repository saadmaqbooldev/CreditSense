from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class ApplicantSector(str, Enum):
    TEXTILE = "textile"
    TRADING = "trading"
    FOOD = "food"
    MANUFACTURING = "manufacturing"
    SERVICES = "services"
    AGRICULTURE = "agriculture"
    TECHNOLOGY = "technology"


@dataclass(frozen=True)
class SectorProfile:
    name: str
    revenue_mean: float
    revenue_std: float
    leverage_mean: float
    leverage_std: float
    growth_mean: float
    margin_mean: float
    default_rate: float


SECTOR_PROFILES: dict[str, SectorProfile] = {
    ApplicantSector.TEXTILE.value: SectorProfile(
        name=ApplicantSector.TEXTILE.value,
        revenue_mean=15000000,
        revenue_std=8000000,
        leverage_mean=1.9,
        leverage_std=0.8,
        growth_mean=0.14,
        margin_mean=0.12,
        default_rate=0.18,
    ),
    ApplicantSector.TRADING.value: SectorProfile(
        name=ApplicantSector.TRADING.value,
        revenue_mean=12000000,
        revenue_std=7000000,
        leverage_mean=2.2,
        leverage_std=0.9,
        growth_mean=0.10,
        margin_mean=0.08,
        default_rate=0.22,
    ),
    ApplicantSector.FOOD.value: SectorProfile(
        name=ApplicantSector.FOOD.value,
        revenue_mean=11000000,
        revenue_std=6000000,
        leverage_mean=1.5,
        leverage_std=0.7,
        growth_mean=0.12,
        margin_mean=0.15,
        default_rate=0.17,
    ),
    ApplicantSector.MANUFACTURING.value: SectorProfile(
        name=ApplicantSector.MANUFACTURING.value,
        revenue_mean=18000000,
        revenue_std=9000000,
        leverage_mean=1.8,
        leverage_std=0.75,
        growth_mean=0.16,
        margin_mean=0.14,
        default_rate=0.20,
    ),
    ApplicantSector.SERVICES.value: SectorProfile(
        name=ApplicantSector.SERVICES.value,
        revenue_mean=9000000,
        revenue_std=5000000,
        leverage_mean=1.2,
        leverage_std=0.6,
        growth_mean=0.18,
        margin_mean=0.20,
        default_rate=0.12,
    ),
    ApplicantSector.AGRICULTURE.value: SectorProfile(
        name=ApplicantSector.AGRICULTURE.value,
        revenue_mean=7000000,
        revenue_std=4500000,
        leverage_mean=2.4,
        leverage_std=1.0,
        growth_mean=0.09,
        margin_mean=0.10,
        default_rate=0.25,
    ),
    ApplicantSector.TECHNOLOGY.value: SectorProfile(
        name=ApplicantSector.TECHNOLOGY.value,
        revenue_mean=22000000,
        revenue_std=12000000,
        leverage_mean=0.9,
        leverage_std=0.4,
        growth_mean=0.28,
        margin_mean=0.24,
        default_rate=0.11,
    ),
}


def _sector_weighted_choice(rng: random.Random) -> str:
    weights = {
        ApplicantSector.TEXTILE.value: 0.18,
        ApplicantSector.TRADING.value: 0.17,
        ApplicantSector.FOOD.value: 0.15,
        ApplicantSector.MANUFACTURING.value: 0.17,
        ApplicantSector.SERVICES.value: 0.15,
        ApplicantSector.AGRICULTURE.value: 0.08,
        ApplicantSector.TECHNOLOGY.value: 0.10,
    }
    return rng.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]


def generate_applicants(n: int = 1000, seed: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []

    for i in range(1, n + 1):
        sector = _sector_weighted_choice(rng)
        profile = SECTOR_PROFILES[sector]

        annual_revenue = max(500000.0, rng.gauss(profile.revenue_mean, profile.revenue_std))
        debt_to_equity = max(0.1, rng.gauss(profile.leverage_mean, profile.leverage_std))
        revenue_growth = max(-0.15, min(0.70, rng.gauss(profile.growth_mean, 0.07)))
        operating_margin = max(0.0, min(0.60, rng.gauss(profile.margin_mean, 0.08)))
        years_in_business = max(1, int(rng.gauss(7, 4)))
        late_payments = max(0, int(rng.gauss(1.5, 2)))
        collateral_coverage = max(0.2, min(3.0, rng.gauss(1.6, 0.5)))

        risk_signal = (
            0.55 * (debt_to_equity / 3.0)
            + 0.20 * max(0.0, 0.2 - operating_margin)
            + 0.15 * max(0.0, 0.5 - collateral_coverage / 2.0)
            + 0.10 * min(1.0, late_payments / 6)
            + 0.15 * max(0.0, -revenue_growth)
        )
        base_probability = min(0.98, max(0.02, profile.default_rate * 0.7 + risk_signal))
        default_probability = min(0.98, max(0.02, base_probability))

        credit_limit_requested = max(
            100000.0,
            annual_revenue * (0.2 + max(0.0, 0.5 - debt_to_equity / 5.0) + max(0.0, revenue_growth) * 0.7),
        )

        record = {
            "applicant_id": f"APP-{i:06d}",
            "sector": sector,
            "annual_revenue": round(annual_revenue, 2),
            "debt_to_equity": round(debt_to_equity, 4),
            "revenue_growth_yoy": round(revenue_growth, 4),
            "operating_margin": round(operating_margin, 4),
            "years_in_business": years_in_business,
            "late_payment_count": late_payments,
            "collateral_coverage_ratio": round(collateral_coverage, 4),
            "credit_limit_requested": round(credit_limit_requested, 2),
            "default_probability": round(default_probability, 6),
            "source": "synthetic",
        }
        records.append(record)

    return records
