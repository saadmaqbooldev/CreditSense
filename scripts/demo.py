#!/usr/bin/env python
"""
CreditSense Demo Script — Interactive walkthrough of underwriting scenarios.

Usage:
    python scripts/demo.py

This script demonstrates:
1. Low-risk compliant applicant (approve)
2. High-leverage applicant (escalate)
3. Regulatory retrieval in action
4. Audit trail inspection
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.orchestration import orchestrate_underwriting
from app.evaluation import RetrievalEvaluator, RETRIEVAL_TEST_CASES
from app.mcp_server import get_audit_log, clear_audit_log
import structlog

logger = structlog.get_logger(__name__)


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_applicant_input(data: dict[str, Any]) -> None:
    print("📋 Applicant Input:")
    for key, value in data.items():
        if key != "applicant_id":
            print(f"   {key:.<30} {value}")


def print_result(result: dict[str, Any]) -> None:
    print("\n✅ Orchestration Result:")
    print(f"\n📊 Financial Analysis:")
    print(f"   Revenue:       ${result['financial_analysis']['normalized_revenue']:,.0f}")
    print(f"   Debt-to-Equity: {result['financial_analysis']['normalized_debt_to_equity']:.2f}")

    print(f"\n📈 Risk Assessment:")
    print(f"   Risk Band:     {result['risk_assessment']['risk_band'].upper()}")
    print(f"   Default Prob:  {result['risk_assessment']['default_probability']:.2%}")
    print(f"   Narrative:     {result['risk_assessment']['narrative'][:100]}...")

    print(f"\n⚖️  Compliance Check:")
    flags_count = len(result["compliance"]["flags"])
    print(f"   Compliance Flags: {flags_count}")
    if flags_count > 0:
        for i, flag in enumerate(result["compliance"]["flags"][:3], 1):
            print(f"     {i}. {flag['rule_id']}: {flag['title']}")
            print(f"        Severity: {flag['severity']}")
            print(f"        Rationale: {flag['rationale'][:60]}...")

    print(f"\n🎯 Final Decision:")
    decision = result["final_decision"]
    status = "✅ APPROVED" if decision["approved"] else "❌ DENIED/ESCALATED"
    print(f"   Status:        {status}")
    print(f"   Escalation:    {'Yes' if decision['escalation_required'] else 'No'}")
    print(f"   Rationale:     {decision['rationale'][:100]}...")
    print(f"   Compliance Flags: {decision['compliance_flags_count']}")


def scenario_1_low_risk_compliant() -> dict[str, Any]:
    """Scenario 1: Well-qualified applicant with strong financials."""
    print_section("Scenario 1: Low-Risk, Compliant Applicant (Expected: APPROVE)")

    applicant_data = {
        "applicant_id": "DEMO-LOW-001",
        "annual_revenue": 25000000.0,
        "debt_to_equity": 0.8,
        "revenue_growth_yoy": 0.15,
        "operating_margin": 0.18,
        "years_in_business": 10,
        "late_payment_count": 0,
        "collateral_coverage_ratio": 1.5,
        "notes": "Profitable manufacturing firm with excellent collateral position and zero delinquencies.",
    }

    print_applicant_input(applicant_data)
    result = orchestrate_underwriting(applicant_data, applicant_data["applicant_id"])
    print_result(result)

    return result


def scenario_2_high_leverage() -> dict[str, Any]:
    """Scenario 2: High-leverage applicant with multiple compliance issues."""
    print_section("Scenario 2: High-Leverage Applicant (Expected: ESCALATE)")

    applicant_data = {
        "applicant_id": "DEMO-HIGH-001",
        "annual_revenue": 10000000.0,
        "debt_to_equity": 3.5,
        "revenue_growth_yoy": 0.05,
        "operating_margin": 0.08,
        "years_in_business": 2,
        "late_payment_count": 4,
        "collateral_coverage_ratio": 0.7,
        "notes": "Highly leveraged startup with weak collateral and persistent delinquency issues.",
    }

    print_applicant_input(applicant_data)
    result = orchestrate_underwriting(applicant_data, applicant_data["applicant_id"])
    print_result(result)

    return result


def scenario_3_retrieval_demo() -> None:
    """Scenario 3: Demonstrate regulation retrieval quality."""
    print_section("Scenario 3: Regulation Retrieval Evaluation")

    evaluator = RetrievalEvaluator()
    batch_result = evaluator.evaluate_batch(RETRIEVAL_TEST_CASES, k=5)

    print(f"📚 Retrieval Quality Metrics (over {batch_result['test_count']} test queries):")
    print(f"   Average Precision@5: {batch_result['avg_precision_at_k']:.2%}")
    print(f"   Average Recall@5:    {batch_result['avg_recall_at_k']:.2%}")

    print(f"\n📋 Sample Query Results:")
    for i, result in enumerate(batch_result["results"][:3], 1):
        print(f"\n   Query {i}: '{result['query'][:40]}...'")
        print(f"      Expected: {', '.join(result['expected_regulations'])}")
        print(f"      Retrieved: {', '.join(result['retrieved_regulations'][:2])}")
        print(f"      Precision: {result['precision_at_k']:.2%}, Recall: {result['recall_at_k']:.2%}")


def audit_trail_demo() -> None:
    """Scenario 4: Inspect audit trails."""
    print_section("Scenario 4: Audit Trail Inspection")

    clear_audit_log()

    applicant_data = {
        "applicant_id": "DEMO-AUDIT-001",
        "annual_revenue": 15000000.0,
        "debt_to_equity": 1.5,
        "revenue_growth_yoy": 0.10,
        "operating_margin": 0.12,
        "years_in_business": 5,
        "late_payment_count": 1,
        "collateral_coverage_ratio": 1.2,
    }

    result = orchestrate_underwriting(applicant_data, applicant_data["applicant_id"])
    audit = get_audit_log()

    print(f"📋 Orchestration for {applicant_data['applicant_id']} completed.")
    print(f"   Decision: {'✅ APPROVED' if result['final_decision']['approved'] else '❌ ESCALATED'}")
    print(f"\n🔍 Audit Log ({len(audit)} entries):")

    for i, entry in enumerate(audit[-5:], 1):
        print(f"\n   Entry {i}:")
        print(f"      Tool: {entry.tool_name}")
        print(f"      Identity: {entry.identity}")
        print(f"      Scope: {entry.scope}")
        print(f"      Status: {entry.status}")
        print(f"      Timestamp: {entry.timestamp}")
        print(f"      Payload Hash: {entry.payload_hash}")


def main() -> None:
    print("\n" + "🎯 " * 35)
    print("  CreditSense Demo — Interactive Underwriting Walkthrough")
    print("🎯 " * 35)

    try:
        result1 = scenario_1_low_risk_compliant()
        result2 = scenario_2_high_leverage()
        scenario_3_retrieval_demo()
        audit_trail_demo()

        print_section("Demo Complete")
        print("""
✅ All scenarios executed successfully!

Key Takeaways:
1. Low-risk applicants with zero compliance flags can be approved automatically.
2. High-leverage applicants with compliance issues are automatically escalated.
3. Regulatory clauses are retrieved with high precision and recall.
4. All underwriting decisions are fully audited with caller identity and timestamp.

For live demo with a running FastAPI server, use:
  curl -X POST http://localhost:8000/submit/underwriting \\
    -H "Content-Type: application/json" \\
    -d @scenario.json
        """)

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
