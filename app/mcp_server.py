from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from app.config import get_settings
from app.rag_pipeline import build_regulation_chunks, retrieve_relevant_chunks
from app.ml_pipeline import score_applicant


logger = structlog.get_logger(__name__)


@dataclass
class AuditEntry:
    timestamp: str
    tool_name: str
    identity: str
    payload_hash: str
    scope: str
    status: str


AUDIT_LOG: list[AuditEntry] = []


def hash_payload(payload: dict[str, Any]) -> str:
    """Create a deterministic hash of a payload for audit trails."""
    payload_json = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload_json.encode()).hexdigest()[:16]


def mask_sensitive_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Mask account numbers and other sensitive fields."""
    masked = record.copy()
    sensitive_fields = ["account_number", "ssn", "tax_id", "bank_account"]
    for field in sensitive_fields:
        if field in masked:
            masked[field] = "***MASKED***"
    return masked


def log_audit_entry(tool_name: str, identity: str, payload: dict[str, Any], scope: str, status: str = "success") -> None:
    """Log tool access for audit trail."""
    entry = AuditEntry(
        timestamp=datetime.utcnow().isoformat(),
        tool_name=tool_name,
        identity=identity,
        payload_hash=hash_payload(payload),
        scope=scope,
        status=status,
    )
    AUDIT_LOG.append(entry)
    logger.info("audit_log", tool=tool_name, identity=identity, scope=scope, status=status)


def check_auth(api_key: str | None, required_scope: str) -> tuple[bool, str]:
    """Check API key and scope authorization."""
    if not api_key:
        return False, "Missing API key"

    api_config = {
        "read-only-key-1": ["read:applicants", "read:regulations"],
        "read-write-key-1": ["read:applicants", "read:regulations", "write:decisions"],
        "admin-key-1": ["*"],
    }

    scopes = api_config.get(api_key, [])
    if not scopes:
        return False, "Invalid API key"

    if "*" in scopes or required_scope in scopes:
        return True, ""

    return False, f"Insufficient scope: {required_scope}"


def get_applicant_financials(applicant_id: str, identity: str, api_key: str | None) -> dict[str, Any]:
    """Tool: Retrieve normalized financial data for an applicant."""
    auth_ok, auth_msg = check_auth(api_key, "read:applicants")
    if not auth_ok:
        log_audit_entry("get_applicant_financials", identity, {"applicant_id": applicant_id}, "read:applicants", "denied")
        return {"error": auth_msg}

    payload = {"applicant_id": applicant_id}
    log_audit_entry("get_applicant_financials", identity, payload, "read:applicants")

    financials = {
        "applicant_id": applicant_id,
        "annual_revenue": 15000000.0,
        "debt_to_equity": 1.8,
        "revenue_growth_yoy": 0.10,
        "operating_margin": 0.15,
        "years_in_business": 6,
        "late_payment_count": 1,
        "collateral_coverage_ratio": 1.3,
        "sector": "manufacturing",
    }

    return mask_sensitive_fields(financials)


def search_sbp_regulations(topic: str, regulation_number: str | None, identity: str, api_key: str | None) -> dict[str, Any]:
    """Tool: Search SBP regulation clauses by topic or regulation number."""
    auth_ok, auth_msg = check_auth(api_key, "read:regulations")
    if not auth_ok:
        log_audit_entry("search_sbp_regulations", identity, {"topic": topic}, "read:regulations", "denied")
        return {"error": auth_msg}

    payload = {"topic": topic, "regulation_number": regulation_number}
    log_audit_entry("search_sbp_regulations", identity, payload, "read:regulations")

    chunks = build_regulation_chunks()
    query = f"{topic} {regulation_number or ''}".strip()
    matches = retrieve_relevant_chunks(query, chunks)

    return {
        "query": query,
        "results": matches,
        "count": len(matches),
    }


def get_portfolio_exposure(sector: str, identity: str, api_key: str | None) -> dict[str, Any]:
    """Tool: Get aggregate sector exposure as a percentage."""
    auth_ok, auth_msg = check_auth(api_key, "read:applicants")
    if not auth_ok:
        log_audit_entry("get_portfolio_exposure", identity, {"sector": sector}, "read:applicants", "denied")
        return {"error": auth_msg}

    payload = {"sector": sector}
    log_audit_entry("get_portfolio_exposure", identity, payload, "read:applicants")

    sector_exposure = {
        "manufacturing": 0.28,
        "retail": 0.22,
        "agriculture": 0.18,
        "services": 0.20,
        "other": 0.12,
    }

    return {
        "sector": sector,
        "exposure_pct": sector_exposure.get(sector.lower(), 0.0),
        "exposure_limit_pct": 0.35,
    }


def submit_underwriting_decision(
    applicant_id: str,
    decision: str,
    rationale: str,
    identity: str,
    api_key: str | None,
) -> dict[str, Any]:
    """Tool: Submit and record an underwriting decision with full audit."""
    auth_ok, auth_msg = check_auth(api_key, "write:decisions")
    if not auth_ok:
        log_audit_entry("submit_underwriting_decision", identity, {"applicant_id": applicant_id}, "write:decisions", "denied")
        return {"error": auth_msg}

    payload = {"applicant_id": applicant_id, "decision": decision, "rationale": rationale}
    log_audit_entry("submit_underwriting_decision", identity, payload, "write:decisions")

    if decision not in {"approve", "deny", "escalate"}:
        return {"error": f"Invalid decision: {decision}"}

    return {
        "applicant_id": applicant_id,
        "decision": decision,
        "rationale": rationale,
        "recorded_by": identity,
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_mcp_tool_definitions() -> list[dict[str, Any]]:
    """Get all MCP tool definitions for registration with the server."""
    return [
        {
            "name": "get_applicant_financials",
            "description": "Retrieve normalized financial data for a specific applicant (read-scoped).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "applicant_id": {"type": "string", "description": "The applicant ID"},
                    "api_key": {"type": "string", "description": "API key for authentication"},
                    "identity": {"type": "string", "description": "Caller identity for audit logging"},
                },
                "required": ["applicant_id", "identity"],
            },
        },
        {
            "name": "search_sbp_regulations",
            "description": "Search SBP regulation clauses by topic or regulation number (read-scoped).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Search topic (e.g., 'leverage', 'collateral')"},
                    "regulation_number": {"type": "string", "description": "Optional regulation number (e.g., 'R-12')"},
                    "api_key": {"type": "string", "description": "API key for authentication"},
                    "identity": {"type": "string", "description": "Caller identity for audit logging"},
                },
                "required": ["topic", "identity"],
            },
        },
        {
            "name": "get_portfolio_exposure",
            "description": "Get aggregate sector exposure as a percentage of portfolio (read-scoped).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sector": {"type": "string", "description": "Sector name (e.g., 'manufacturing', 'agriculture')"},
                    "api_key": {"type": "string", "description": "API key for authentication"},
                    "identity": {"type": "string", "description": "Caller identity for audit logging"},
                },
                "required": ["sector", "identity"],
            },
        },
        {
            "name": "submit_underwriting_decision",
            "description": "Submit an underwriting decision with full audit trail (write-scoped).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "applicant_id": {"type": "string", "description": "The applicant ID"},
                    "decision": {"type": "string", "enum": ["approve", "deny", "escalate"], "description": "Decision: approve, deny, or escalate"},
                    "rationale": {"type": "string", "description": "Explanation of the decision"},
                    "api_key": {"type": "string", "description": "API key for authentication"},
                    "identity": {"type": "string", "description": "Caller identity for audit logging"},
                },
                "required": ["applicant_id", "decision", "rationale", "identity"],
            },
        },
    ]


def call_mcp_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch tool calls to the appropriate handler."""
    identity = arguments.get("identity", "unknown")
    api_key = arguments.get("api_key")

    if name == "get_applicant_financials":
        return get_applicant_financials(arguments["applicant_id"], identity, api_key)

    elif name == "search_sbp_regulations":
        return search_sbp_regulations(arguments["topic"], arguments.get("regulation_number"), identity, api_key)

    elif name == "get_portfolio_exposure":
        return get_portfolio_exposure(arguments["sector"], identity, api_key)

    elif name == "submit_underwriting_decision":
        return submit_underwriting_decision(
            arguments["applicant_id"],
            arguments["decision"],
            arguments["rationale"],
            identity,
            api_key,
        )

    else:
        return {"error": f"Unknown tool: {name}"}


def get_audit_log() -> list[AuditEntry]:
    """Return the full audit log for testing and inspection."""
    return AUDIT_LOG.copy()


def clear_audit_log() -> None:
    """Clear the audit log (for testing purposes)."""
    global AUDIT_LOG
    AUDIT_LOG = []
