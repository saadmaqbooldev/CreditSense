# CreditSense — Interview Q&A (90-Second Answers)

> Prepared responses for the four interview questions in Section 7 of the master tracker.

---

## Question 1: Guardrail Example

**Q**: *What guardrail did you build that **prevents** a wrong agent action, with a concrete example? (Reference: `UnderwritingSupervisorAgent` hard-approval rule.)*

**A** (90 seconds):

The `UnderwritingSupervisorAgent` enforces a hard approval guardrail in executable code—not as a soft LLM suggestion. Approval requires **both**:
- Risk score < 0.6 (not high), **AND**
- Zero unresolved compliance flags

This is a boolean check. If an applicant has even one compliance flag—say, debt-to-equity > 2.0 triggering the R-12 leverage rule—the decision automatically becomes `approved: false` and `escalation_required: true`, regardless of how low their default probability is.

**Concrete example**: An applicant with `debt_to_equity=3.5` and `default_probability=0.35` (which would normally be "low risk") is rejected because the leverage violation is not negotiable. You cannot bypass this via prompt injection or adversarial input because the logic is hardcoded: if `len(compliance_flags) > 0`, approval is impossible. This ensures regulatory violations block approval every time.

---

## Question 2: Deterministic Agents vs. LLM Frameworks

**Q**: *Why use deterministic agents instead of a heavyweight framework like LangGraph or CrewAI?*

**A** (90 seconds):

Three critical reasons for underwriting:

1. **Auditability & Reproducibility**: Every agent step is deterministic. Run the same applicant data 100 times, get the same result. With LLM frameworks, hallucinations and inconsistency are harder to rule out. Regulators need reproducible, auditable decisions.

2. **Guardrail Enforcement**: Hard rules (like "no approval if compliance flags exist") are trivial to implement in code and impossible to bypass via prompt injection. With LLMs, guardrails are suggestions in system prompts—a determined attacker can jailbreak them.

3. **Test Coverage & Confidence**: We achieved 50 passing tests with zero flakiness. Unit tests can validate every agent path without mocking LLM responses. For less safety-critical tasks, LLM-powered agents are fine. For credit underwriting with regulatory guardrails and audit trails, deterministic logic is safer.

The trade-off is that deterministic agents require explicit logic for each decision rule, but this is appropriate for underwriting where every decision must be citable and reproducible.

---

## Question 3: Citation-Backed Compliance

**Q**: *How does the compliance agent avoid making unsupported claims?*

**A** (90 seconds):

The `ComplianceAgent` enforces a "no claim without source" rule. Every compliance flag **must** include a `source` field pointing to the retrieved regulation chunk. Here's how it works:

1. **Trigger a Rule**: If debt-to-equity > 2.0, the leverage rule is triggered.
2. **Retrieve Clauses**: The agent runs a RAG query for "leverage debt equity" and retrieves candidate SBP regulation chunks.
3. **Attach Citation**: The best-matching clause (e.g., R-12: "Maximum Leverage and Debt-to-Equity Risk Controls") is attached to the flag.
4. **Escalate if No Source**: If no regulation can be found supporting the flag, the system escalates to a human rather than guessing.

Result: Every flag is citable. An auditor can ask "why did you flag this?" and we reply "because R-12 states..." with the exact regulation text. This makes compliance decisions transparent, traceable, and defensible.

---

## Question 4: Orchestration & Error Handling

**Q**: *How does the system handle errors during underwriting?*

**A** (90 seconds):

The `orchestrate_underwriting()` function implements a three-tier error strategy:

1. **Transient Errors (Auto-Retry)**: Network glitches, timeouts, temporary service unavailability? Retry up to 2 times. Each retry logs the attempt.

2. **Hard Errors (Immediate Escalation)**: Input validation failures, malformed compliance data? Escalate immediately with full error context and audit trail intact.

3. **Exhaustion Escalation**: After 2 retries, if still failing, raise `OrchestrationError` with decision rationale, logs, and instrumentation so debugging is rapid.

**Structured logging** (via structlog) captures applicant_id, stage (financial_analysis, risk_scoring, compliance, supervision), and status at every step. Example:

```json
{
  "event": "orchestration_start",
  "applicant_id": "APP-001",
  "retry_count": 0
}
{
  "event": "compliance_check_complete",
  "applicant_id": "APP-001",
  "flags_count": 2
}
{
  "event": "supervision_complete",
  "applicant_id": "APP-001",
  "approved": false
}
```

This enables rapid debugging: you can trace exactly which stage failed and with what context.

---

## Bonus: System Architecture Overview

If asked "Tell me about the system architecture in 3 minutes":

**Three-Layer Design**:

1. **Scoring Layer** (Phase 3–4):
   - Baseline ML model (logistic-style scoring) for default probability
   - Hybrid RAG retrieval (BM25 + dense candidates, reranked top-5) for regulation clauses
   - 10 SBP regulation topics in corpus; precision@5 and recall@5 both > 0.8

2. **Decision Layer** (Phase 5):
   - Four deterministic agents (Financial Analyst, Risk Scorer, Compliance Checker, Supervisor)
   - Supervisor enforces hard guardrail: approve only if risk < 0.6 AND zero flags
   - Every compliance flag cites a source regulation

3. **Integration & Audit Layer** (Phase 6–7):
   - FastAPI endpoint `/submit/underwriting` chains all agents
   - MCP server exposes scoped read/write tools (applicant data, regulations, portfolio exposure, decision submission)
   - Audit trail logs identity, timestamp, payload hash for every tool call
   - Middleware enforces rate limiting (60 req/min) and structured logging

**Testing**: 50 passing tests across all layers. Adversarial tests verify guardrails cannot be bypassed.

---

## Interview Tips

- **Lead with the guardrail**: It's the most differentiating feature. Most underwriting systems lack hard-coded safety.
- **Emphasize reproducibility**: "Run it 100 times, same result" is a massive advantage over LLMs.
- **Cite concrete examples**: Instead of "we validate applicants," say "an applicant with debt-to-equity=3.5 is automatically escalated even if default_probability=0.35."
- **Show the tests**: "50 passing tests, zero flakiness" is proof of robustness. Be ready to walk through one test case.
- **Mention the audit trail**: "Every decision is logged with caller identity and timestamp" shows regulatory readiness.

---

**Last Updated**: 2026-08-29  
**Status**: Interview-ready
