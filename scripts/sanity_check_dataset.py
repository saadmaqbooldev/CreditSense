"""Run distribution and data-quality checks on the final synthetic dataset."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "raw" / "creditsense_final_portfolio_v1.csv"
DEFAULT_REPORT_PATH = BASE_DIR / "scripts" / "outputs" / "final_dataset_sanity_report.json"

EXPECTED_SECTORS = {
    "textile",
    "trading",
    "food",
    "manufacturing",
    "services",
    "agriculture",
    "technology",
}


def parse_number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", value)
    if not cleaned:
        return None
    decimal_index = cleaned.rfind(".")
    integer_part = cleaned[:decimal_index].replace(".", "")
    fractional_part = cleaned[decimal_index + 1:]
    normalized = f"{integer_part}.{fractional_part}" if decimal_index >= 0 else integer_part
    return float(normalized)


def numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "std": round(pstdev(values), 4),
    }


def run_sanity_check(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    with dataset_path.open(newline="", encoding="utf-8") as dataset_file:
        records = list(csv.DictReader(dataset_file))

    sector_counts = Counter(record["sector"] for record in records)
    numeric_fields = (
        "annual_revenue",
        "debt_to_equity",
        "revenue_growth_yoy",
        "operating_margin",
        "collateral_coverage_ratio",
        "default_probability",
    )
    numeric_by_sector: dict[str, dict[str, dict[str, float]]] = {}
    for sector in sorted(sector_counts):
        sector_records = [record for record in records if record["sector"] == sector]
        numeric_by_sector[sector] = {
            field: numeric_summary(
                [parsed for record in sector_records if (parsed := parse_number(record[field])) is not None]
            )
            for field in numeric_fields
            if any(parse_number(record[field]) is not None for record in sector_records)
        }

    missing_collateral = sum(
        1 for record in records if record["collateral_coverage_ratio"] in ("", None)
    )
    notes_present = sum(1 for record in records if record["credit_officer_notes"].strip())
    default_probabilities = [
        parsed
        for record in records
        if (parsed := parse_number(record["default_probability"])) is not None
    ]
    checks = {
        "row_count_10000_to_20000": 10_000 <= len(records) <= 20_000,
        "all_expected_sectors_present": set(sector_counts) == EXPECTED_SECTORS,
        "sector_counts_nonzero": all(count > 0 for count in sector_counts.values()),
        "notes_complete": notes_present == len(records),
        "default_probability_range_valid": all(0.0 <= value <= 1.0 for value in default_probabilities),
        "collateral_missing_rate_near_5_percent": 0.03 <= missing_collateral / len(records) <= 0.07,
    }
    report = {
        "dataset": str(dataset_path),
        "row_count": len(records),
        "sector_counts": dict(sorted(sector_counts.items())),
        "sector_shares": {
            sector: round(count / len(records), 4)
            for sector, count in sorted(sector_counts.items())
        },
        "numeric_summary_by_sector": numeric_by_sector,
        "missing_collateral_count": missing_collateral,
        "missing_collateral_rate": round(missing_collateral / len(records), 4),
        "notes_present": notes_present,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = run_sanity_check()
    print(json.dumps(report["checks"], indent=2))
    print(f"Wrote sanity report: {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()
