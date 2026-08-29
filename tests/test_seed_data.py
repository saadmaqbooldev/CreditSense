import pytest
from sqlalchemy import text

from app.db import SessionLocal
from app.models import Applicant
from app.seed_data import seed_applicants


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
