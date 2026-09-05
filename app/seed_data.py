from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data_generation import generate_applicants
from app.db import SessionLocal
from app.models import Applicant

BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_DATASET_PATH = BASE_DIR / "data" / "raw" / "creditsense_final_portfolio_v1.csv"


def _parse_number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", value)
    if not cleaned:
        return None
    decimal_index = cleaned.rfind(".")
    integer_part = cleaned[:decimal_index].replace(".", "")
    fractional_part = cleaned[decimal_index + 1:]
    normalized = f"{integer_part}.{fractional_part}" if decimal_index >= 0 else integer_part
    return float(normalized)


def load_dataset_records(dataset_path: Path = FINAL_DATASET_PATH) -> list[dict[str, Any]]:
    """Load and normalize the versioned CSV for database persistence."""
    with dataset_path.open(newline="", encoding="utf-8") as dataset_file:
        rows = list(csv.DictReader(dataset_file))

    required_fields = {"applicant_id", "sector", "annual_revenue"}
    if not rows or not required_fields.issubset(rows[0]):
        raise ValueError(f"Dataset must contain fields: {sorted(required_fields)}")

    records = []
    for row in rows:
        annual_revenue = _parse_number(row["annual_revenue"])
        if annual_revenue is None:
            raise ValueError(f"Missing annual revenue for {row['applicant_id']}")
        records.append(
            {
                "applicant_id": row["applicant_id"],
                "sector": row["sector"],
                "annual_revenue": annual_revenue,
                "debt_to_equity": _parse_number(row.get("debt_to_equity")),
                "revenue_growth_yoy": _parse_number(row.get("revenue_growth_yoy")),
                "operating_margin": _parse_number(row.get("operating_margin")),
                "years_in_business": int(row["years_in_business"]) if row.get("years_in_business") else None,
                "late_payment_count": int(row["late_payment_count"]) if row.get("late_payment_count") else None,
                "collateral_coverage_ratio": _parse_number(row.get("collateral_coverage_ratio")),
                "credit_limit_requested": _parse_number(row.get("credit_limit_requested")),
                "default_probability": _parse_number(row.get("default_probability")),
                "credit_officer_notes": row.get("credit_officer_notes") or None,
                "source": row.get("source") or "synthetic",
            }
        )
    return records


def _add_applicant(session: Session, record: dict[str, Any]) -> bool:
    existing = session.execute(
        select(Applicant).where(Applicant.applicant_id == record["applicant_id"])
    ).scalar_one_or_none()
    if existing is not None:
        return False

    session.add(Applicant(**record))
    return True


def seed_applicants(session: Session, n: int = 1000, seed: int = 42) -> int:
    records = generate_applicants(n=n, seed=seed)

    inserted = sum(_add_applicant(session, record) for record in records)

    session.commit()
    return inserted


def seed_dataset(session: Session, dataset_path: Path = FINAL_DATASET_PATH) -> int:
    """Persist a CSV dataset and return the number of newly inserted applicants."""
    records = load_dataset_records(dataset_path)
    inserted = sum(_add_applicant(session, record) for record in records)
    session.commit()
    return inserted


def seed_demo_data() -> int:
    with SessionLocal() as session:
        return seed_applicants(session, n=1000, seed=42)


def seed_final_dataset() -> int:
    with SessionLocal() as session:
        return seed_dataset(session)


if __name__ == "__main__":
    count = seed_final_dataset()
    print(f"Seeded {count} final-dataset applicants into the database.")
