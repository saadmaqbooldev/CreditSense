"""add sector portfolio exposure view

Revision ID: 20260905_000003
Revises: 20260905_000002
"""

from alembic import op
import sqlalchemy as sa


revision = "20260905_000003"
down_revision = "20260905_000002"
branch_labels = None
depends_on = None


VIEW_NAME = "sector_portfolio_exposure"


def upgrade() -> None:
    op.execute(
        sa.text(
            f"""
            CREATE VIEW {VIEW_NAME} AS
            SELECT
                sector,
                COUNT(*) AS applicant_count,
                COALESCE(SUM(credit_limit_requested), 0) AS total_requested_exposure,
                COALESCE(AVG(credit_limit_requested), 0) AS average_requested_exposure,
                COALESCE(AVG(default_probability), 0) AS average_default_probability,
                COUNT(*) FILTER (WHERE collateral_coverage_ratio IS NULL)
                    AS missing_collateral_count,
                CASE
                    WHEN NULLIF(SUM(SUM(credit_limit_requested)) OVER (), 0) IS NULL THEN 0
                    ELSE SUM(credit_limit_requested)
                        / NULLIF(SUM(SUM(credit_limit_requested)) OVER (), 0)
                END AS exposure_share
            FROM applicants
            GROUP BY sector
            """
        )
    )


def downgrade() -> None:
    op.execute(sa.text(f"DROP VIEW IF EXISTS {VIEW_NAME}"))
