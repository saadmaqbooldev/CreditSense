from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data_generation import generate_applicants
from app.db import SessionLocal
from app.models import Applicant


def seed_applicants(session: Session, n: int = 1000, seed: int = 42) -> int:
    records = generate_applicants(n=n, seed=seed)

    for record in records:
        existing = session.execute(
            select(Applicant).where(Applicant.applicant_id == record["applicant_id"])
        ).scalar_one_or_none()

        if existing is not None:
            continue

        applicant = Applicant(
            applicant_id=record["applicant_id"],
            sector=record["sector"],
            annual_revenue=record["annual_revenue"],
            debt_to_equity=record["debt_to_equity"],
            revenue_growth_yoy=record["revenue_growth_yoy"],
            operating_margin=record["operating_margin"],
            years_in_business=record["years_in_business"],
            late_payment_count=record["late_payment_count"],
            collateral_coverage_ratio=record["collateral_coverage_ratio"],
            credit_limit_requested=record["credit_limit_requested"],
            default_probability=record["default_probability"],
            source=record["source"],
        )
        session.add(applicant)

    session.commit()
    return len(records)


def seed_demo_data() -> int:
    with SessionLocal() as session:
        return seed_applicants(session, n=1000, seed=42)


if __name__ == "__main__":
    count = seed_demo_data()
    print(f"Seeded {count} applicants into the database.")
