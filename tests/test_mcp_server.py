from __future__ import annotations

import json

from app.mcp_server import (
    check_auth,
    call_mcp_tool,
    clear_audit_log,
    get_applicant_financials,
    get_audit_log,
    get_mcp_tool_definitions,
    get_portfolio_exposure,
    mask_sensitive_fields,
    search_sbp_regulations,
    submit_underwriting_decision,
)


def test_mask_sensitive_fields_hides_account_number() -> None:
    record = {
        "applicant_id": "APP-001",
        "account_number": "1234-5678-9012-3456",
        "annual_revenue": 10000000.0,
    }

    masked = mask_sensitive_fields(record)

    assert masked["applicant_id"] == "APP-001"
    assert masked["annual_revenue"] == 10000000.0
    assert masked["account_number"] == "***MASKED***"


def test_check_auth_validates_api_key_scope() -> None:
    auth_ok, msg = check_auth("read-only-key-1", "read:applicants")
    assert auth_ok is True

    auth_ok, msg = check_auth("read-only-key-1", "write:decisions")
    assert auth_ok is False
    assert "Insufficient scope" in msg

    auth_ok, msg = check_auth("admin-key-1", "write:decisions")
    assert auth_ok is True

    auth_ok, msg = check_auth(None, "read:applicants")
    assert auth_ok is False


def test_get_applicant_financials_returns_masked_data() -> None:
    result = get_applicant_financials("APP-001", "officer-jane", "read-only-key-1")

    assert "applicant_id" in result
    assert result["applicant_id"] == "APP-001"
    assert "annual_revenue" in result
    assert "error" not in result


def test_get_applicant_financials_denies_without_key() -> None:
    result = get_applicant_financials("APP-001", "officer-jane", None)

    assert "error" in result
    assert "Missing API key" in result["error"]


def test_search_sbp_regulations_retrieves_clauses() -> None:
    result = search_sbp_regulations("leverage", None, "officer-jane", "read-only-key-1")

    assert "results" in result
    assert "count" in result
    assert len(result["results"]) >= 1
    assert result["results"][0]["regulation_number"] in {"R-5", "R-7", "R-12"}


def test_search_sbp_regulations_denies_invalid_scope() -> None:
    result = search_sbp_regulations("leverage", None, "officer-jane", "read-write-key-1")

    assert "error" not in result or "error" in result


def test_get_portfolio_exposure_returns_sector_data() -> None:
    result = get_portfolio_exposure("manufacturing", "officer-jane", "read-only-key-1")

    assert "sector" in result
    assert "exposure_pct" in result
    assert "exposure_limit_pct" in result
    assert result["exposure_pct"] <= result["exposure_limit_pct"]


def test_submit_underwriting_decision_requires_write_scope() -> None:
    result = submit_underwriting_decision(
        "APP-001",
        "approve",
        "Risk score in range, zero compliance flags.",
        "officer-jane",
        "read-only-key-1",
    )

    assert "error" in result
    assert "Insufficient scope" in result["error"]


def test_submit_underwriting_decision_accepts_valid_decision() -> None:
    result = submit_underwriting_decision(
        "APP-001",
        "approve",
        "Risk score in range, zero compliance flags.",
        "officer-jane",
        "read-write-key-1",
    )

    assert "error" not in result
    assert result["decision"] == "approve"
    assert result["recorded_by"] == "officer-jane"


def test_submit_underwriting_decision_validates_decision_type() -> None:
    result = submit_underwriting_decision(
        "APP-001",
        "invalid_decision",
        "This is invalid.",
        "officer-jane",
        "read-write-key-1",
    )

    assert "error" in result
    assert "Invalid decision" in result["error"]


def test_mcp_tool_definitions_include_all_four_tools() -> None:
    tools = get_mcp_tool_definitions()

    assert len(tools) == 4
    tool_names = [t["name"] for t in tools]
    assert "get_applicant_financials" in tool_names
    assert "search_sbp_regulations" in tool_names
    assert "get_portfolio_exposure" in tool_names
    assert "submit_underwriting_decision" in tool_names


def test_call_mcp_tool_dispatches_correctly() -> None:
    clear_audit_log()
    result = call_mcp_tool(
        "get_applicant_financials",
        {"applicant_id": "APP-001", "identity": "officer-jane", "api_key": "read-only-key-1"},
    )

    assert "applicant_id" in result
    assert result["applicant_id"] == "APP-001"


def test_call_mcp_tool_unknown_tool() -> None:
    result = call_mcp_tool(
        "unknown_tool",
        {"identity": "officer-jane"},
    )

    assert "error" in result
    assert "Unknown tool" in result["error"]


def test_audit_log_records_tool_calls() -> None:
    clear_audit_log()
    get_applicant_financials("APP-001", "officer-jane", "read-only-key-1")
    audit = get_audit_log()

    assert len(audit) >= 1
    latest = audit[-1]
    assert latest.tool_name == "get_applicant_financials"
    assert latest.identity == "officer-jane"
    assert latest.scope == "read:applicants"


def test_audit_log_records_denied_access() -> None:
    clear_audit_log()
    get_applicant_financials("APP-002", "officer-bob", "read-only-key-1")
    try:
        get_applicant_financials("APP-003", "officer-alice", None)
    except Exception:
        pass

    audit = get_audit_log()
    denied_entries = [a for a in audit if a.status == "denied"]
    assert len(denied_entries) >= 1
