"""add applicant notes

Revision ID: 20260905_000002
Revises: 20260829_000001
"""

from alembic import op
import sqlalchemy as sa


revision = "20260905_000002"
down_revision = "20260829_000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("applicants", sa.Column("credit_officer_notes", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("applicants", "credit_officer_notes")