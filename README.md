 # CreditSense

CreditSense is a deterministic SME underwriting assistant for Pakistani fintech workflows. It combines a calibrated XGBoost risk model, SBP-style regulation retrieval, compliance checks, and a hard underwriting guardrail.

## Quick Start

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Generate and Train

```powershell
& ".\.venv\Scripts\python.exe" scripts/generate_final_dataset.py
& ".\.venv\Scripts\python.exe" scripts/train_xgboost_baseline.py
& ".\.venv\Scripts\python.exe" scripts/train_xgboost_calibrated.py
& ".\.venv\Scripts\python.exe" scripts/evaluate_xgboost_model.py
& ".\.venv\Scripts\python.exe" scripts/fairness_drift_monitor.py
```

The final dataset is written to `data/raw/creditsense_final_portfolio_v1.csv` with sibling schema metadata. Model, evaluation, TreeSHAP, fairness, and drift artifacts are written to `app/ml_artifacts/`.

## API Endpoints

- `GET /health` checks service status.
- `POST /predict/credit-risk` validates an applicant and returns probability, risk band, and model version.
- `POST /submit/underwriting` runs normalization, risk scoring, regulation retrieval, compliance checks, and the supervisor decision.

Example risk request:

```json
{
	"annual_revenue": 18000000,
	"debt_to_equity": 1.4,
	"revenue_growth_yoy": 0.12,
	"operating_margin": 0.18,
	"years_in_business": 7,
	"late_payment_count": 1,
	"collateral_coverage_ratio": 1.5
}
```

## Architecture

1. `app.data_generation` creates sector-conditioned applicants and optional messy source fields.
2. `app.ml_pipeline` engineers features and exposes the API scoring contract.
3. `scripts/train_xgboost_calibrated.py` trains constrained XGBoost and applies Platt scaling.
4. `app.rag_pipeline` retrieves citable regulation chunks.
5. `app.agents` and `app.orchestration` apply deterministic compliance and approval guardrails.
6. `app.mcp_server` exposes scoped, audited tools with field masking.

## Database

PostgreSQL is optional for local development. Apply migrations and seed the final dataset when PostgreSQL is running:

```powershell
alembic upgrade head
& ".\.venv\Scripts\python.exe" -m app.seed_data
```

The `sector_portfolio_exposure` view aggregates requested exposure by sector.

## Testing

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
```

Database-backed tests skip gracefully when PostgreSQL is unavailable.
