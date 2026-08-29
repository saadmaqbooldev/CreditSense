"""init schema

Revision ID: 20260829_000001
Revises: 
Create Date: 2026-08-29 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260829_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "applicants",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("applicant_id", sa.String(length=64), nullable=False),
        sa.Column("sector", sa.String(length=64), nullable=False),
        sa.Column("annual_revenue", sa.Float(), nullable=False),
        sa.Column("debt_to_equity", sa.Float(), nullable=True),
        sa.Column("revenue_growth_yoy", sa.Float(), nullable=True),
        sa.Column("operating_margin", sa.Float(), nullable=True),
        sa.Column("years_in_business", sa.Integer(), nullable=True),
        sa.Column("late_payment_count", sa.Integer(), nullable=True),
        sa.Column("collateral_coverage_ratio", sa.Float(), nullable=True),
        sa.Column("credit_limit_requested", sa.Float(), nullable=True),
        sa.Column("default_probability", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=True),
        sa.UniqueConstraint("applicant_id"),
    )
    op.create_index(op.f("ix_applicants_id"), "applicants", ["id"], unique=False)
    op.create_index(op.f("ix_applicants_applicant_id"), "applicants", ["applicant_id"], unique=False)

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("document_type", sa.String(length=64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.UniqueConstraint("document_id"),
    )
    op.create_index(op.f("ix_documents_id"), "documents", ["id"], unique=False)
    op.create_index(op.f("ix_documents_document_id"), "documents", ["document_id"], unique=False)

    op.create_table(
        "regulation_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("regulation_number", sa.String(length=64), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.UniqueConstraint("chunk_id"),
    )
    op.create_index(op.f("ix_regulation_chunks_id"), "regulation_chunks", ["id"], unique=False)
    op.create_index(op.f("ix_regulation_chunks_chunk_id"), "regulation_chunks", ["chunk_id"], unique=False)

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=True),
        sa.Column("payload", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_index(op.f("ix_audit_logs_id"), "audit_logs", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_audit_logs_id"), table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index(op.f("ix_regulation_chunks_chunk_id"), table_name="regulation_chunks")
    op.drop_index(op.f("ix_regulation_chunks_id"), table_name="regulation_chunks")
    op.drop_table("regulation_chunks")
    op.drop_index(op.f("ix_documents_document_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_id"), table_name="documents")
    op.drop_table("documents")
    op.drop_index(op.f("ix_applicants_applicant_id"), table_name="applicants")
    op.drop_index(op.f("ix_applicants_id"), table_name="applicants")
    op.drop_table("applicants")
