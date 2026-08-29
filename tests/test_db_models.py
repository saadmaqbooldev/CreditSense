from app.db import Base
from app.models import Applicant, AuditLog, Document, RegulationChunk


def test_database_models_are_registered() -> None:
    model_names = set(Base.metadata.tables)

    assert "applicants" in model_names
    assert "documents" in model_names
    assert "regulation_chunks" in model_names
    assert "audit_logs" in model_names

    applicant_table = Base.metadata.tables["applicants"]
    assert "applicant_id" in applicant_table.columns
    assert "sector" in applicant_table.columns
    assert "annual_revenue" in applicant_table.columns

    chunk_table = Base.metadata.tables["regulation_chunks"]
    assert "content" in chunk_table.columns
    assert "embedding" in chunk_table.columns


def test_applicant_model_columns_are_expected() -> None:
    applicant_columns = {column.name for column in Applicant.__table__.columns}

    assert {"id", "applicant_id", "sector", "annual_revenue", "default_probability"}.issubset(applicant_columns)
    assert Applicant.__table__.name == "applicants"
    assert RegulationChunk.__table__.name == "regulation_chunks"
