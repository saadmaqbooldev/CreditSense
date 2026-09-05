import pytest
from sqlalchemy import text

from app.db import SessionLocal
from app.models import Applicant
from app.seed_data import load_dataset_records, seed_applicants


def test_load_dataset_records_normalizes_final_csv() -> None:
    records = load_dataset_records()

    assert len(records) == 15_000
    assert records[0]["applicant_id"].startswith("APP-")
    assert isinstance(records[0]["annual_revenue"], float)
    assert any(record["collateral_coverage_ratio"] is None for record in records)
    assert all(record["credit_officer_notes"] for record in records)


def test_seed_applicants_persists_records() -> None:
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("Postgres is not running locally; DB-backed seed check is skipped until the service is started.")

    with SessionLocal() as session:
        session.query(Applicant).delete()
        session.commit()

        inserted = seed_applicants(session, n=25, seed=7)

        count = session.query(Applicant).count()
        assert inserted == 25
        assert count == 25
        assert session.query(Applicant).first().applicant_id.startswith("APP-")

        session.query(Applicant).delete()
        session.commit()
