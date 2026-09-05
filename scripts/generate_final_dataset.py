"""Generate the versioned final synthetic applicant dataset."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "raw"
DEFAULT_ROW_COUNT = 15_000
DEFAULT_SEED = 42
DATASET_VERSION = "1.0.0"
DATASET_NAME = "creditsense_final_portfolio"

sys.path.insert(0, str(BASE_DIR))
from app.data_generation import generate_applicants  # noqa: E402


def _serialize_record(record: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(record)
    serialized["data_quality_flags"] = "|".join(record.get("data_quality_flags", []))
    return serialized


def _schema(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for field in records[0]:
        values = [record[field] for record in records]
        non_null_values = [value for value in values if value is not None]
        sample = non_null_values[0] if non_null_values else None
        fields[field] = {
            "type": "string" if isinstance(sample, str) else "number",
            "nullable": any(value is None for value in values),
        }
    return fields


def generate_final_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    row_count: int = DEFAULT_ROW_COUNT,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, Path]:
    """Generate the final CSV and its schema/version metadata file."""
    if not 10_000 <= row_count <= 20_000:
        raise ValueError("row_count must be between 10,000 and 20,000")

    records = generate_applicants(n=row_count, seed=seed, messy=True)
    serialized_records = [_serialize_record(record) for record in records]
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{DATASET_NAME}_v{DATASET_VERSION.split('.')[0]}.csv"
    metadata_path = csv_path.with_suffix(".meta.json")
    fieldnames = list(serialized_records[0])

    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serialized_records)

    metadata = {
        "dataset_name": DATASET_NAME,
        "version": DATASET_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(records),
        "seed": seed,
        "generator": "app.data_generation.generate_applicants",
        "messy_mode": True,
        "quality_rates": {
            "ocr_noise_rate": 0.10,
            "currency_format_rate": 0.25,
            "missing_collateral_rate": 0.05,
        },
        "schema": _schema(records),
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return csv_path, metadata_path


def main() -> None:
    csv_path, metadata_path = generate_final_dataset()
    print(f"Wrote dataset: {csv_path}")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
