# CreditSense — SME Credit Risk & SBP Regulatory Compliance Assistant

A production-grade underwriting assistant that scores SME credit risk with a baseline gradient-boosted-style model, retrieves relevant SBP prudential regulations via hybrid RAG, and enforces hard guardrails to ensure every approval is compliant.

## Quick Start

### Prerequisites
- Python 3.11+
- FastAPI, SQLAlchemy, PostgreSQL (optional for full DB features)
- See `requirements.txt` for complete dependencies

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the App

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### Key Endpoints

#### Health Check
```bash
GET /health
```

#### Quick Credit Risk Prediction (baseline model only)
```bash
POST /predict/credit-risk
Content-Type: application/json

{
  "annual_revenue": 15000000,
  "debt_to_equity": 1.5,
  "revenue_growth_yoy": 0.12,
  "operating_margin": 0.15,
  "years_in_business": 6,
  "late_payment_count": 1,
  "collateral_coverage_ratio": 1.3
}
```

#### Full Orchestrated Underwriting (agents + guardrails)
```bash
POST /submit/underwriting
Content-Type: application/json

{
  "applicant_id": "APP-2026-0001",
  "annual_revenue": 15000000,
  "debt_to_equity": 1.5,
  "revenue_growth_yoy": 0.12,
  "operating_margin": 0.15,
  "years_in_business": 6,
  "late_payment_count": 1,
  "collateral_coverage_ratio": 1.3,
  "notes": "Established retail business with seasonal cash flow patterns."
}
```

Returns a structured decision with:
- Financial analysis (normalized inputs, raw notes)
- Risk assessment (default probability, risk band, narrative)
- Compliance findings (flags, source citations)
- Final decision (approved/denied/escalated, rationale)

## Architecture

### Core Components

**Phase 0–2**: Project scaffold, applicant data generation, database schema  
**Phase 3**: Baseline ML scoring model (logistic-style feature engineering)  
**Phase 4**: RAG retrieval pipeline with SBP regulation corpus  
**Phase 5**: AI agents (FinancialAnalystAgent, RiskScoringAgent, ComplianceAgent, UnderwritingSupervisorAgent)  
**Phase 6**: MCP server with scoped read/write tools, field masking, audit logging  
**Phase 7**: Orchestration engine with retry logic and structured logging  
**Phase 8**: Comprehensive evaluation suite (50+ tests, guardrail validation, retrieval metrics)  

### Key Design Decisions

1. **Lightweight Agent Pattern**: Uses deterministic dataclass-based agents instead of a heavyweight LLM framework, ensuring reproducibility and test coverage.

2. **Hard Guardrails in Code**: The `UnderwritingSupervisorAgent` enforces approval logic as executable code, not as a soft prompt. Approval requires:
   - Risk score < 0.6 (not high)
   - Zero unresolved compliance flags
   
   This ensures the guardrail cannot be bypassed via prompt injection.

3. **Citation-Backed Compliance**: Every compliance flag includes a citation to the source SBP regulation chunk. If the system cannot find a regulation supporting a flag, it escalates rather than guessing.

4. **Audit Trail on Every Tool Call**: All MCP tool calls are logged with identity, timestamp, and payload hash. This enables forensic review of every decision.

5. **Retry Logic with Exhaustion**: Transient errors trigger up to 2 automatic retries. Hard errors escalate immediately with full context.

## Testing & Evaluation

Run the full test suite:

```bash
pytest -q
```

Expected: 50 passing tests, 1 skipped (DB seeding skipped if Postgres unavailable).

### Test Coverage

- **Unit Tests**: ML scoring, RAG retrieval, agent behavior, MCP tools
- **Integration Tests**: End-to-end applicant processing, multi-profile scenarios
- **Guardrail Tests**: Adversarial attempts to approve with compliance flags all rejected
- **Retrieval Evaluation**: Precision@5 and recall@5 across 10 SBP regulation topics

## Demo Walkthrough

### Scenario 1: Low-Risk, Compliant Applicant (Expected: Approve)

```bash
curl -X POST http://localhost:8000/submit/underwriting \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_id": "DEMO-LOW-001",
    "annual_revenue": 25000000,
    "debt_to_equity": 0.8,
    "revenue_growth_yoy": 0.15,
    "operating_margin": 0.18,
    "years_in_business": 10,
    "late_payment_count": 0,
    "collateral_coverage_ratio": 1.5,
    "notes": "Profitable manufacturing firm, strong collateral position."
  }'
```

**Expected Output**: 
- Risk band: likely moderate or low
- Zero compliance flags
- `approved: true` (if risk < 0.6)

### Scenario 2: High-Leverage Applicant (Expected: Escalate)

```bash
curl -X POST http://localhost:8000/submit/underwriting \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_id": "DEMO-HIGH-001",
    "annual_revenue": 10000000,
    "debt_to_equity": 3.5,
    "revenue_growth_yoy": 0.05,
    "operating_margin": 0.08,
    "years_in_business": 2,
    "late_payment_count": 4,
    "collateral_coverage_ratio": 0.7,
    "notes": "Highly leveraged with weak collateral and delinquency issues."
  }'
```

**Expected Output**:
- Risk band: high
- Multiple compliance flags (leverage, delinquency, collateral)
- Each flag cites a regulation source
- `approved: false`, `escalation_required: true`

### Scenario 3: Regulatory Clause Retrieval

```bash
curl -X POST http://localhost:8000/submit/underwriting \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_id": "DEMO-RAG-001",
    "annual_revenue": 18000000,
    "debt_to_equity": 2.2,
    "revenue_growth_yoy": 0.10,
    "operating_margin": 0.12,
    "years_in_business": 5,
    "late_payment_count": 2,
    "collateral_coverage_ratio": 1.1,
    "notes": "Sector exposure concerns, monitor concentration."
  }'
```

**Expected Output**:
- ComplianceAgent retrieves R-7 (Sectoral Exposure clause)
- Flags are generated with R-7 as source
- Decision rationale cites the regulation

## MCP Server & Integration

The project includes a Python MCP server (`app/mcp_server.py`) that exposes:

- `get_applicant_financials(applicant_id)` — Read applicant data (scoped read)
- `search_sbp_regulations(topic)` — Search regulatory clauses (scoped read)
- `get_portfolio_exposure(sector)` — Check sector concentration (scoped read)
- `submit_underwriting_decision(applicant_id, decision, rationale)` — Record decision (scoped write)

All tool calls require valid API keys with appropriate scopes. Sensitive fields are masked before output.

## Future Enhancements (Beyond Phase 8)

- **Real XGBoost Model**: Phase 3 baseline can be upgraded to a full gradient-boosted model trained on real SME portfolios.
- **Dense Embeddings + pgvector**: Phase 4 retrieval can integrate dense embeddings for semantic search.
- **LLM-Powered Agents**: Phase 5 agents can be upgraded to use Claude or GPT-4 via LangGraph for more sophisticated reasoning.
- **Postgres Backend**: Full database integration for applicant storage, decision audit trails, and portfolio analytics.
- **Fairness & Drift Monitoring**: Phase 3 can add demographic parity checks and model drift detection.

## Interview Q&A

### Question 1: Hard Guardrail Example

**Q**: What guardrail did you build that *prevents* a wrong agent action, with a concrete example?

**A**: The `UnderwritingSupervisorAgent` enforces a hard approval guardrail in executable code. Approval requires *both* risk score < 0.6 (not high) *and* zero unresolved compliance flags. If an applicant has even one compliance flag (e.g., debt-to-equity > 2.0 triggering the leverage rule), the decision is automatically `approved: false` and `escalation_required: true`. This is not a soft LLM suggestion—it's a boolean check in the code. Example: an applicant with `debt_to_equity=3.5` and `default_probability=0.35` would be rejected because the leverage violation is not negotiable, even though the default probability is technically "low-moderate".

### Question 2: Why Determinstic Agents Over LLM Frameworks

**Q**: Why use deterministic agents instead of LangGraph or CrewAI?

**A**: Deterministic agents (implemented as dataclasses) offer three critical advantages for underwriting:

1. **Auditability**: Every decision step is deterministic, reproducible, and testable. With LLM frameworks, hallucinations and inconsistency are harder to rule out.
2. **Guardrail Enforcement**: Hard rules (like "no approval if compliance flags exist") are trivial to implement in code and impossible to bypass via prompt injection.
3. **Test Coverage**: Unit tests can validate every agent path without mocking LLM responses. We achieved 50 passing tests with zero flakiness.

For less safety-critical tasks, LLM-powered agents are appropriate. For credit underwriting with regulatory guardrails, deterministic logic is safer.

### Question 3: Retrieval Citation Pattern

**Q**: How does the compliance agent avoid making unsupported claims?

**A**: The `ComplianceAgent` enforces a "no claim without source" rule. Every compliance flag includes a `source` field pointing to the retrieved regulation chunk. If a rule is triggered (e.g., debt-to-equity > 2.0), the agent retrieves relevant clauses via hybrid RAG and attaches the best match to the flag. If no regulation can be found supporting the flag, the system escalates to a human rather than guessing. This ensures every compliance decision is citable and auditable.

### Question 4: Orchestration & Retry Pattern

**Q**: How does the system handle errors during underwriting?

**A**: The `orchestrate_underwriting()` function implements a three-tier error strategy:

1. **Transient Errors**: Retries up to 2 times with full context logging.
2. **Hard Errors**: Escalates immediately with error details and instrumentation.
3. **Escalation**: On repeated failure, raises `OrchestrationError` with decision rationale and audit trail intact.

Structured logging (via structlog) captures applicant_id, stage (financial_analysis, risk_scoring, compliance, supervision), and status at every step. This enables rapid debugging of production issues.

## Project Metrics

- **Test Coverage**: 50 passing tests across 8 phases
- **Retrieval Quality**: Precision@5 and recall@5 measured across 10 SBP regulation topics
- **Guardrail Enforcement**: 100% pass rate on adversarial guardrail bypass attempts
- **Code Quality**: Linted with Ruff, type-checked with Pylance

## File Structure

```
CreditSense/
├── app/
│   ├── main.py                  # FastAPI app and endpoints
│   ├── config.py                # Settings and environment
│   ├── data_generation.py        # Synthetic applicant generator
│   ├── db.py                    # Database engine
│   ├── models.py                # SQLAlchemy ORM models
│   ├── ml_pipeline.py           # Scoring logic
│   ├── rag_pipeline.py          # Retrieval and regulation corpus
│   ├── agents.py                # Orchestration agents
│   ├── mcp_server.py            # MCP tool definitions
│   ├── orchestration.py         # Full underwriting flow
│   ├── middleware.py            # Rate limiting and logging
│   └── evaluation.py            # Retrieval evaluator
├── tests/
│   ├── test_health.py           # Health endpoint
│   ├── test_data_generation.py  # Applicant generator
│   ├── test_db_models.py        # ORM models
│   ├── test_ml_pipeline.py      # Scoring logic
│   ├── test_rag_pipeline.py     # Retrieval
│   ├── test_agents.py           # Agent behavior
│   ├── test_mcp_server.py       # MCP tools and auth
│   ├── test_orchestration.py    # Full flow
│   └── test_phase8_evaluation.py # Comprehensive evaluation
├── migrations/                   # Alembic DB migrations
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Local Postgres setup
├── Dockerfile                   # App container image
├── alembic.ini                 # Migration config
├── .env.example                # Environment template
└── CreditSense_Master_Tracker.md # Project timeline and decisions
```

## Contact & Support

For questions about architecture, guardrails, or retrieval strategy, refer to the tracker and test suite as the source of truth. All design decisions are documented with rationale.

---

**Last Updated**: 2026-08-29  
**Status**: Phases 0–8 complete (Phase 9: Demo & Docs in progress)
