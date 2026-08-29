from __future__ import annotations

from typing import Any


REGULATION_SEED: list[dict[str, str]] = [
    {
        "regulation_number": "R-5",
        "title": "Exposure Limits and Single Borrower Concentration",
        "content": "Banks and fintech lenders must not exceed single-borrower exposure limits for a given borrower or related group. Exposure limits apply across both funded and unfunded commitments and should reflect aggregate risk concentration."
    },
    {
        "regulation_number": "R-7",
        "title": "Sectoral Exposure and Risk Diversification",
        "content": "Sector concentration should be assessed across agriculture, manufacturing, consumer retail, and services. Institutions must monitor sectoral exposure relative to capital buffer and may require additional provisioning when sector stress is elevated."
    },
    {
        "regulation_number": "R-9",
        "title": "Collateral Valuation Standards",
        "content": "Collateral valuation must reflect prudent market values and may not be overstated. Where collateral coverage is below minimum thresholds, lenders must escalate or restructure the facility."
    },
    {
        "regulation_number": "R-12",
        "title": "Maximum Leverage and Debt-to-Equity Risk Controls",
        "content": "Debt-to-equity ratios above policy thresholds may require additional approval, especially for trading and agriculture borrowers. Excessive leverage without sufficient cash flow support is treated as a risk concern."
    },
    {
        "regulation_number": "R-14",
        "title": "Late Payment and Delinquency Reporting",
        "content": "Late payment counts and delinquency trends should be monitored and reported. Persistent delinquency or adverse trends require human review and may trigger restructuring or provisioning actions."
    },
    {
        "regulation_number": "R-18",
        "title": "SME Cash Flow Assessment",
        "content": "SME credit approvals should rely on cash flow realism and repayment capacity. Revenue growth and operating margin should be evaluated in context of working capital stress and seasonal cyclicality."
    },
    {
        "regulation_number": "R-21",
        "title": "Underwriting Governance and Decision Auditability",
        "content": "All underwriting decisions must be supported by a documented rationale and traceable evidence. Final decisions must not rely on unsupported assumptions or hidden policy overrides."
    },
]


def build_regulation_chunks() -> list[dict[str, str]]:
    return [
        {
            "regulation_number": row["regulation_number"],
            "title": row["title"],
            "content": row["content"],
        }
        for row in REGULATION_SEED
    ]


def retrieve_relevant_chunks(query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_words = set(query.lower().replace("-", " ").split())

    scored: list[dict[str, Any]] = []
    for chunk in chunks:
        content = (chunk.get("title", "") + " " + chunk.get("content", "")).lower()
        score = sum(2 for word in query_words if word in content)
        score += sum(1 for phrase in ["exposure", "borrower", "leverage", "sector", "collateral", "late payment", "cash flow"] if phrase in content)
        scored.append({**chunk, "score": score})

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:5]
