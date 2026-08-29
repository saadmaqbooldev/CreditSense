# CreditSense — SME Credit Risk & SBP Regulatory Compliance Assistant
### Master Project Context & Progress Tracker

> **This file is the single source of truth for this project.** It is meant to be pasted or uploaded into any AI coding assistant (Claude, ChatGPT, Gemini, Cursor, etc.) at the start of a session so the assistant has full context — what the system is, what's been built, how it was built, and what's still open — without the human needing to re-explain progress verbally every time.

---

## 0. Instructions for the AI Assistant Reading This File

Read this whole file before answering. Specifically:

1. **Treat the status flags in Section 4 as ground truth.** Do not assume a task is incomplete just because it "sounds hard," and do not re-explain or redo anything flagged `[x] DONE` unless the user explicitly asks for a change/refactor.
2. **Read the "Notes" line under each task before helping with it.** That line records the actual implementation choice made (library, pattern, schema decision, gotcha). Stay consistent with it instead of proposing a different approach from scratch — unless the current approach is genuinely broken, in which case flag the concern before changing course.
3. **Section 3 ("Fixed Technical Decisions") is binding.** Don't suggest swapping the agreed stack (e.g., "use OpenAI instead of Claude," "use Chroma instead of pgvector") unless asked to reconsider it.
4. **Work on the next `[ ]` or `[~]` item in the current phase** unless the user directs you elsewhere. Don't jump ahead to later phases out of enthusiasm.
5. **When you finish helping with a task, remind the user to update its status flag and Notes line** — you cannot edit this file yourself in most tool contexts, so hand that back to the user explicitly.
6. If something in this doc looks stale or contradicted by the actual code the user shares, say so — don't silently trust the doc over the ground truth of the repo.

---

## 1. Project Concept (Context, Not a To-Do)

**Title:** CreditSense — Automated SME Underwriting & Prudential Regulation Assistant

**Business problem:** Pakistani fintechs and digital lending arms of banks serving SME clients need to underwrite loans quickly while staying compliant with State Bank of Pakistan (SBP) prudential regulations. Credit officers currently rely on spreadsheet-based scoring and manual cross-checks of loan structuring against SBP circulars — slow, error-prone, and a hard cap on how many SME applications can be processed per week.

**What the system does:** Scores SME credit risk with a gradient-boosted model trained on financial-statement and behavioral features, while a compliance agent checks proposed loan terms against SBP prudential regulations via hybrid RAG retrieval — flagging violations before they reach a human approver.

**Interview pitch (for reference, not implementation):** "We built an underwriting copilot that scores SME credit risk with a gradient-boosted model trained on financial-statement and behavioral features, while a compliance agent checks proposed loan terms against SBP prudential regulations using hybrid RAG retrieval — flagging violations before they reach a human approver. For a digital lender processing 500+ SME applications a month, this cuts credit-officer review time significantly and reduces regulatory exposure — a P&L and risk-committee concern, not just a UX nicety."

### System Flow
Applicant documents/financials uploaded → FastAPI ingestion → `FinancialAnalystAgent` normalizes data → XGBoost model scores default risk → `ComplianceAgent` retrieves relevant SBP clauses for the proposed loan structure → `UnderwritingSupervisorAgent` applies a hard guardrail (approve only if risk score in range **and** zero unresolved compliance flags) → decision + audit trail returned via MCP-exposed tools.

---

## 2. Architecture Reference (Context, Not a To-Do)

### 2.1 Classical ML Layer — XGBoost
- Task: binary classification `default_probability_12m` + regression head `recommended_credit_limit`.
- Features: financial ratios (current ratio, debt-to-equity, revenue growth YoY), sector risk code, years in business, bank statement volatility (std dev of monthly inflows), existing loan exposure, late-payment history count, collateral coverage ratio.
- **Monotonic constraints** on features like debt-to-equity (risk must not decrease as leverage increases) — a real underwriting-model detail, not optional polish.
- **Platt scaling** to calibrate raw XGBoost scores into usable probabilities.

### 2.2 Hybrid RAG Layer
- Corpus: SBP Prudential Regulations for SMEs (public circulars), simulated internal credit policy docs, loan structuring templates.
- Storage: PostgreSQL 16 + `pgvector`, with a `tsvector` column for BM25 alongside the `vector` column for dense embeddings.
- Retrieval: BM25 (`ts_rank_cd`) + dense cosine similarity run in parallel, merged via **Reciprocal Rank Fusion (RRF)**.
- **Chunking by regulation clause number** (e.g., "Regulation R-5: Exposure Limits") — not fixed token windows — so retrieval returns citable, auditable clauses.

### 2.3 AI Agents Layer
- `FinancialAnalystAgent` — extracts/normalizes financial statement data from uploaded documents (must tolerate messy/OCR-style input).
- `RiskScoringAgent` — calls the ML model, translates the score into underwriting narrative.
- `ComplianceAgent` — retrieves relevant SBP regulation clauses for the proposed loan structure and flags breaches (e.g., single-borrower exposure limits). **Must not state a duty/limit without citing a retrieved source chunk** — if it can't find one, it returns "insufficient data, escalate to human."
- `UnderwritingSupervisorAgent` — final decision node. Hard guardrail: approve **only if** risk score is within threshold **and** compliance agent reports zero unresolved flags. This is a hard rule, not a soft LLM suggestion — enforce it in code, not just in a prompt.

### 2.4 Custom MCP Server Layer
Tools to expose:
- `get_applicant_financials(applicant_id)` — scoped read.
- `search_sbp_regulations(topic, regulation_number)` — wraps hybrid RAG retrieval.
- `get_portfolio_exposure(sector)` — used by `ComplianceAgent` to check aggregate sector exposure limits.
- `submit_underwriting_decision(applicant_id, decision, rationale)` — write-scoped, fully audited.

Security requirement: **field-level data masking** in MCP tool responses (e.g., masked account numbers) — a genuine fintech data-security expectation, and a strong interview talking point.

### 2.5 Data Strategy Reference
- ~10,000–20,000 synthetic SME applicant records, with financial-ratio distributions **conditioned on sector** (textile SMEs vs. tech startups vs. trading firms have genuinely different leverage/margin norms — encode the correlation, don't randomize uniformly).
- Injected messiness: ~10% of records simulate OCR-extracted bank statements with digit-transposition noise and inconsistent currency formatting (PKR / Rs. / numeric-only); ~5% missing collateral fields.
- RAG corpus: 200+ chunked clauses from real/simulated SBP-style prudential regulation text, with intentional cross-references between clauses (e.g., Regulation X references Regulation Y) to test multi-hop retrieval.
- Multilingual noise: credit-officer notes with Urdu-English code-switching (e.g., "client ka cash flow seasonal hai, Eid se pehle spike hota hai") to stress-test `FinancialAnalystAgent`'s text parsing.

---

## 3. Fixed Technical Decisions (Binding — Confirm Before Deviating)

> Fill these in as you decide. Once filled, treat as locked unless you explicitly revisit them — this prevents different AI sessions from nudging you toward a different stack each time.

| Decision | Choice | Date Locked | Notes |
|---|---|---|---|
| LLM provider (agents) | _TBD_ | | Blueprint suggests Claude or OpenAI SDK |
| Agent framework | _TBD_ | | Blueprint suggests LangGraph or CrewAI |
| Embedding model | _TBD_ | | Blueprint suggests Anthropic/OpenAI embeddings or `bge-large` |
| Reranker | _TBD_ | | Blueprint suggests `bge-reranker-base` |
| Database hosting | _TBD_ | | Local Docker Postgres vs. managed |
| Deployment target | _TBD_ | | Local demo / cloud VM / other |
| Repo structure convention | _TBD_ | | See Section 6 for proposed default |
| Package/dependency manager | _TBD_ | | e.g., `uv`, `poetry`, `pip + venv` |
| Testing framework | `pytest` | | From tech stack baseline |
| Auth approach for API/MCP | _TBD_ | | API key vs. JWT vs. other |

---

## 4. Implementation Task Tracker

**Status legend** — use these exact tags so any AI tool parses them consistently:
- `[ ] TODO` — not started
- `[~] IN_PROGRESS` — started, not complete
- `[x] DONE` — complete and working
- `[!] BLOCKED` — stuck; reason must go in Notes
- `[s] SKIPPED` — deliberately descoped; reason must go in Notes

Each task has a **Notes** line — record the actual library/pattern used, key decisions, or gotchas once you touch it. Leave placeholders (`—`) until then.

### Phase 0 — Project Scaffolding & Environment
- [ ] TODO **P0.1** — Initialize repo structure (see Section 6 proposed layout)
  - Notes: —
- [ ] TODO **P0.2** — Set up `.env` + config management, `.env.example` committed
  - Notes: —
- [ ] TODO **P0.3** — Docker Compose: Postgres (+ pgvector extension), app service, agent service
  - Notes: —
- [ ] TODO **P0.4** — Base FastAPI app skeleton (health check endpoint, Pydantic v2 settings)
  - Notes: —
- [ ] TODO **P0.5** — CI scaffold (GitHub Actions: lint + test on push)
  - Notes: —

### Phase 1 — Synthetic Data Generation
- [ ] TODO **P1.1** — Design applicant schema (fields, types, sector categories)
  - Notes: —
- [ ] TODO **P1.2** — Build sector-conditioned generator (financial ratios correlate realistically per sector, not uniform random)
  - Notes: —
- [ ] TODO **P1.3** — Inject messiness: OCR-style digit noise (~10%), inconsistent currency formatting, missing collateral fields (~5%)
  - Notes: —
- [ ] TODO **P1.4** — Add Urdu-English code-switched free-text notes field
  - Notes: —
- [ ] TODO **P1.5** — Generate final dataset (target: 10,000–20,000 rows), save with schema/version metadata
  - Notes: —
- [ ] TODO **P1.6** — Sanity-check distributions (per-sector plots/summary stats) to confirm realism
  - Notes: —

### Phase 2 — Database & Infra
- [ ] TODO **P2.1** — Postgres schema: applicants table, documents table (RAG corpus), chunks table (`tsvector` + `vector` columns), audit_log table
  - Notes: —
- [ ] TODO **P2.2** — SQLAlchemy 2.0 models + migrations (Alembic or equivalent)
  - Notes: —
- [ ] TODO **P2.3** — Seed script to load synthetic applicant data into Postgres
  - Notes: —
- [ ] TODO **P2.4** — SQL views for portfolio exposure aggregation (sector-level exposure)
  - Notes: —

### Phase 3 — Classical ML Layer (XGBoost)
- [ ] TODO **P3.1** — EDA + feature engineering pipeline
  - Notes: —
- [ ] TODO **P3.2** — Train/test split, baseline XGBoost model
  - Notes: —
- [ ] TODO **P3.3** — Apply monotonic constraints (e.g., debt-to-equity → risk direction)
  - Notes: —
- [ ] TODO **P3.4** — Platt scaling for probability calibration
  - Notes: —
- [ ] TODO **P3.5** — Cross-validation + feature importance / SHAP explainability
  - Notes: —
- [ ] TODO **P3.6** — Model versioning (MLflow or pickle + metadata JSON)
  - Notes: —
- [ ] TODO **P3.7** — `/predict/credit-risk` FastAPI endpoint (Pydantic-validated request/response, SHAP output included)
  - Notes: —
- [ ] TODO **P3.8** — Unit tests for scoring pipeline
  - Notes: —
- [ ] TODO **P3.9** (stretch) — Bias/fairness sanity check across sectors
  - Notes: —
- [ ] TODO **P3.10** (stretch) — Drift monitoring checks
  - Notes: —

### Phase 4 — RAG Corpus & Hybrid Retrieval
- [ ] TODO **P4.1** — Source/simulate SBP prudential regulation text (target 200+ chunks) with intentional cross-references between clauses
  - Notes: —
- [ ] TODO **P4.2** — Chunk by clause number (not fixed token windows)
  - Notes: —
- [ ] TODO **P4.3** — Embedding generation pipeline, populate `vector` column
  - Notes: —
- [ ] TODO **P4.4** — BM25 setup via `tsvector` / `ts_rank_cd`
  - Notes: —
- [ ] TODO **P4.5** — RRF fusion logic combining BM25 + dense results
  - Notes: —
- [ ] TODO **P4.6** — Reranker integration (e.g., `bge-reranker-base`) on top-N before final top-5
  - Notes: —
- [ ] TODO **P4.7** — Retrieval evaluation harness (precision@5, recall@10) — including multi-hop clause cross-reference cases
  - Notes: —

### Phase 5 — AI Agents Layer
- [ ] TODO **P5.1** — Choose/confirm framework (LangGraph vs CrewAI) — record in Section 3
  - Notes: —
- [ ] TODO **P5.2** — `FinancialAnalystAgent`: parse/normalize uploaded financials, tolerate messy input, parse Urdu-English notes
  - Notes: —
- [ ] TODO **P5.3** — `RiskScoringAgent`: call ML endpoint, produce plain-English risk narrative
  - Notes: —
- [ ] TODO **P5.4** — `ComplianceAgent`: retrieve SBP clauses for proposed loan structure, flag breaches, enforce "no claim without a cited source chunk" rule
  - Notes: —
- [ ] TODO **P5.5** — `UnderwritingSupervisorAgent`: hard-coded guardrail (approve only if risk-in-range AND zero unresolved compliance flags)
  - Notes: —
- [ ] TODO **P5.6** — Conditional routing / escalation-to-human path when confidence is low
  - Notes: —
- [ ] TODO **P5.7** — Structured outputs via Pydantic schemas for all agent responses
  - Notes: —

### Phase 6 — Custom MCP Server
- [ ] TODO **P6.1** — MCP server scaffold (official Python MCP SDK)
  - Notes: —
- [ ] TODO **P6.2** — Tool: `get_applicant_financials(applicant_id)` — scoped read
  - Notes: —
- [ ] TODO **P6.3** — Tool: `search_sbp_regulations(topic, regulation_number)` — wraps hybrid RAG
  - Notes: —
- [ ] TODO **P6.4** — Tool: `get_portfolio_exposure(sector)`
  - Notes: —
- [ ] TODO **P6.5** — Tool: `submit_underwriting_decision(applicant_id, decision, rationale)` — write-scoped, audited
  - Notes: —
- [ ] TODO **P6.6** — Field-level data masking middleware (e.g., account numbers)
  - Notes: —
- [ ] TODO **P6.7** — API-key/scope-based auth middleware for MCP tool access
  - Notes: —
- [ ] TODO **P6.8** — Audit log table wired to every tool call (identity, timestamp, payload hash)
  - Notes: —

### Phase 7 — Backend Integration
- [ ] TODO **P7.1** — Wire ingestion endpoint → agents → ML endpoint → RAG → supervisor decision
  - Notes: —
- [ ] TODO **P7.2** — Error handling: ML service timeouts, malformed LLM outputs, retry logic
  - Notes: —
- [ ] TODO **P7.3** — Structured JSON logging (`structlog` or equivalent)
  - Notes: —
- [ ] TODO **P7.4** — Rate-limiting on write-tools
  - Notes: —

### Phase 8 — Testing & Evaluation
- [ ] TODO **P8.1** — Unit tests across ML, RAG, agents, MCP tools
  - Notes: —
- [ ] TODO **P8.2** — End-to-end integration test (full applicant → decision flow)
  - Notes: —
- [ ] TODO **P8.3** — Guardrail test: confirm supervisor cannot approve with unresolved compliance flags, even under adversarial prompt input
  - Notes: —
- [ ] TODO **P8.4** — Retrieval evaluation numbers recorded (precision@5, recall@10) — not just anecdotal demo success
  - Notes: —

### Phase 9 — Demo, Docs & Interview Readiness
- [ ] TODO **P9.1** — Minimal frontend or Postman collection for live demo
  - Notes: —
- [ ] TODO **P9.2** — Demo script / README walkthrough
  - Notes: —
- [ ] TODO **P9.3** — Decision audit trail dashboard (stretch)
  - Notes: —
- [ ] TODO **P9.4** — Prepare 90-second answers to the 4 interview questions in Section 7
  - Notes: —

---

## 5. Decision Log

> Append-only. One row per non-trivial decision made during the build — especially anywhere you deviated from the blueprint in Section 2.

| Date | Decision | Rationale | Alternatives Considered |
|---|---|---|---|
| _e.g. 2026-08-27_ | _e.g. Used LangGraph over CrewAI_ | _e.g. Better conditional-edge support for hard guardrail node_ | _e.g. CrewAI_ |

---

## 6. Proposed Repo Structure

> Edit to match reality once scaffolding (P0.1) is done — keep this in sync so any AI tool can navigate the repo correctly.

```
creditsense/
├── data/
│   ├── generators/          # synthetic data generation scripts
│   └── raw_corpus/          # SBP regulation source text before chunking
├── db/
│   ├── models.py            # SQLAlchemy models
│   └── migrations/
├── ml/
│   ├── train.py
│   ├── features.py
│   └── artifacts/           # saved model + metadata
├── rag/
│   ├── chunking.py
│   ├── embed.py
│   ├── retrieval.py         # BM25 + dense + RRF + rerank
│   └── eval.py
├── agents/
│   ├── financial_analyst.py
│   ├── risk_scoring.py
│   ├── compliance.py
│   └── supervisor.py
├── mcp_server/
│   ├── server.py
│   ├── tools.py
│   └── auth_middleware.py
├── api/
│   └── main.py               # FastAPI app
├── tests/
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 7. Interview Readiness Checklist (Context Reference)

Prepare a concrete, 90-second answer for each — tied to what was actually built, not the blueprint text:

1. What guardrail did you build that *prevents* a wrong agent action, with a concrete example? (Reference: `UnderwritingSupervisorAgent` hard-approval rule.)
2. What are your retrieval evaluation numbers (precision@k), not just "it works when I demo it"?
3. What real data-quality problem did you simulate, and how did your pipeline handle it? (Reference: OCR noise, currency formatting, Urdu-English code-switching.)
4. What are the system's failure modes — what happens when the ML service times out, or the LLM hallucinates a compliance clause?

---

## 8. Open Questions / Risks

> Running list — add as they come up, remove/resolve as answered.

- —

---

## 9. Changelog of This Tracker File

| Date | Change |
|---|---|
| 2026-08-27 | Initial file created from AI_Capstone_Blueprints_2026.md, Blueprint 2 (CreditSense) |
