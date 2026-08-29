#!/usr/bin/env python
"""
CreditSense Audit Trail Viewer — Inspect tool access and decision logs.

Usage:
    python scripts/audit_viewer.py [filter_tool] [filter_status]

Examples:
    python scripts/audit_viewer.py                                    # Show all
    python scripts/audit_viewer.py submit_underwriting_decision       # Show decision submissions
    python scripts/audit_viewer.py "" denied                          # Show denied access attempts
"""

import sys
from typing import Optional
from app.mcp_server import get_audit_log


def print_audit_table(entries: list, max_rows: Optional[int] = None) -> None:
    """Print audit log entries in a formatted table."""
    if not entries:
        print("No audit entries found.")
        return

    if max_rows:
        entries = entries[-max_rows:]

    header = f"{'Timestamp':<25} {'Tool':<30} {'Identity':<15} {'Scope':<20} {'Status':<8} {'Hash':<8}"
    print("\n" + header)
    print("-" * len(header))

    for entry in entries:
        timestamp = entry.timestamp[-8:] if len(entry.timestamp) > 8 else entry.timestamp
        tool_name = entry.tool_name[:28]
        identity = entry.identity[:13]
        scope = entry.scope[:18]
        status = entry.status[:6]
        payload_hash = entry.payload_hash[:6]

        print(f"{timestamp:<25} {tool_name:<30} {identity:<15} {scope:<20} {status:<8} {payload_hash:<8}")


def filter_audit_log(
    entries: list,
    tool_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
) -> list:
    """Filter audit log by tool name and/or status."""
    filtered = entries

    if tool_filter:
        filtered = [e for e in filtered if tool_filter.lower() in e.tool_name.lower()]

    if status_filter:
        filtered = [e for e in filtered if status_filter.lower() == e.status.lower()]

    return filtered


def print_audit_summary(entries: list) -> None:
    """Print summary statistics of audit log."""
    if not entries:
        print("No audit entries.")
        return

    total = len(entries)
    by_tool = {}
    by_status = {}
    by_identity = {}

    for entry in entries:
        by_tool[entry.tool_name] = by_tool.get(entry.tool_name, 0) + 1
        by_status[entry.status] = by_status.get(entry.status, 0) + 1
        by_identity[entry.identity] = by_identity.get(entry.identity, 0) + 1

    print(f"\n📊 Audit Log Summary ({total} total entries)")
    print(f"\nBy Tool:")
    for tool, count in sorted(by_tool.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tool:<35} {count:>5} calls")

    print(f"\nBy Status:")
    for status, count in sorted(by_status.items(), key=lambda x: x[1], reverse=True):
        print(f"  {status:<35} {count:>5} entries")

    print(f"\nBy Identity:")
    for identity, count in sorted(by_identity.items(), key=lambda x: x[1], reverse=True):
        print(f"  {identity:<35} {count:>5} calls")


def main() -> None:
    tool_filter = sys.argv[1] if len(sys.argv) > 1 else None
    status_filter = sys.argv[2] if len(sys.argv) > 2 else None

    audit_log = get_audit_log()

    print("\n" + "="*80)
    print("  CreditSense Audit Trail Viewer")
    print("="*80)

    print_audit_summary(audit_log)

    filtered = filter_audit_log(audit_log, tool_filter, status_filter)

    filter_str = ""
    if tool_filter or status_filter:
        filters = []
        if tool_filter:
            filters.append(f"tool contains '{tool_filter}'")
        if status_filter:
            filters.append(f"status = '{status_filter}'")
        filter_str = f" (filtered by {', '.join(filters)})"

    print(f"\n📋 Recent Audit Entries{filter_str}:")
    print_audit_table(filtered, max_rows=20)

    print("\n" + "="*80)
    print(f"Total entries shown: {len(filtered)} of {len(audit_log)}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
