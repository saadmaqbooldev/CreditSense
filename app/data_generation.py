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


def _transpose_digits(value: float, rng: random.Random) -> str:
    """Return a monetary value with two adjacent digits transposed."""
    value_text = f"{value:.2f}"
    digit_positions = [index for index, character in enumerate(value_text) if character.isdigit()]
    swap_index = rng.randrange(len(digit_positions) - 1)
    first_position = digit_positions[swap_index]
    second_position = digit_positions[swap_index + 1]
    value_characters = list(value_text)
    value_characters[first_position], value_characters[second_position] = (
        value_characters[second_position],
        value_characters[first_position],
    )
    return "".join(value_characters)


def _format_currency(value: float, rng: random.Random) -> str:
    """Return a monetary value in one of the source-format variants."""
    formats = (
        "PKR {value:,.2f}",
        "Rs. {value:,.2f}",
        "{value:,.2f}",
        "PKR {value:.2f}",
    )
    return rng.choice(formats).format(value=value)


def _credit_officer_note(
    sector: str,
    debt_to_equity: float,
    late_payments: int,
    revenue_growth: float,
    rng: random.Random,
) -> str:
    """Create a deterministic Roman Urdu-English underwriting note."""
    if late_payments >= 3 or debt_to_equity >= 2.5:
        templates = (
            "Client ka cash flow pressure mein hai; repayment history needs close review.",
            "Business ka leverage high hai aur late payments hain; collateral verification required.",
        )
    elif revenue_growth < 0:
        templates = (
            "Client ki sales decline ho rahi hain; working capital plan needs validation.",
            "Revenue growth negative hai, is liye bank statement trend review karna hoga.",
        )
    else:
        templates = (
            "Client ka cash flow stable hai; bank statements support the requested facility.",
            "Business ki performance theek hai; routine documentation and sector checks required.",
        )

    return f"{rng.choice(templates)} Sector: {sector}."


def generate_applicants(
    n: int = 1000,
    seed: int | None = None,
    *,
    messy: bool = False,
    ocr_noise_rate: float = 0.10,
    currency_format_rate: float = 0.25,
    missing_collateral_rate: float = 0.05,
) -> list[dict]:
    """Generate sector-conditioned applicants with optional source-data messiness.

    Messy mode simulates OCR digit transposition, mixed PKR/Rs. formatting, and
    missing collateral fields while retaining flags for downstream auditing.
    """
    for rate_name, rate in (
        ("ocr_noise_rate", ocr_noise_rate),
        ("currency_format_rate", currency_format_rate),
        ("missing_collateral_rate", missing_collateral_rate),
    ):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{rate_name} must be between 0 and 1")

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
            "credit_officer_notes": _credit_officer_note(
                sector,
                debt_to_equity,
                late_payments,
                revenue_growth,
                rng,
            ),
            "source": "synthetic",
        }

        if messy:
            quality_flags: list[str] = []
            monetary_fields = ("annual_revenue", "credit_limit_requested")

            if rng.random() < ocr_noise_rate:
                field = rng.choice(monetary_fields)
                record[field] = _transpose_digits(record[field], rng)
                quality_flags.append(f"ocr_digit_noise:{field}")

            for field in monetary_fields:
                if rng.random() < currency_format_rate and isinstance(record[field], float):
                    record[field] = _format_currency(record[field], rng)
                    quality_flags.append(f"currency_format:{field}")

            if rng.random() < missing_collateral_rate:
                record["collateral_coverage_ratio"] = None
                quality_flags.append("missing_collateral")

            record["data_quality_flags"] = quality_flags
        records.append(record)

    return records
