import json

from app.data_generation import ApplicantSector, generate_applicants
from scripts.generate_final_dataset import generate_final_dataset
from scripts.sanity_check_dataset import run_sanity_check


def test_generate_applicants_returns_expected_shape() -> None:
    records = generate_applicants(250, seed=42)

    assert len(records) == 250
    assert all("applicant_id" in record for record in records)
    assert all("sector" in record for record in records)
    assert all("annual_revenue" in record for record in records)
    assert all("debt_to_equity" in record for record in records)
    assert all("credit_limit_requested" in record for record in records)

    sectors = {record["sector"] for record in records}
    assert sectors.issubset(set(item.value for item in ApplicantSector))

    assert all(0.0 <= record["default_probability"] <= 1.0 for record in records)
    assert all(record["credit_limit_requested"] > 0 for record in records)


def test_generate_applicants_messy_mode_injects_source_data_issues() -> None:
    records = generate_applicants(2000, seed=42, messy=True)

    ocr_records = [
        record for record in records
        if any(flag.startswith("ocr_digit_noise:") for flag in record["data_quality_flags"])
    ]
    formatted_values = [
        record[field]
        for record in records
        for field in ("annual_revenue", "credit_limit_requested")
        if isinstance(record[field], str)
    ]
    missing_collateral = [
        record for record in records
        if record["collateral_coverage_ratio"] is None
    ]

    assert 0.05 <= len(ocr_records) / len(records) <= 0.15
    assert formatted_values
    assert any(value.startswith(("PKR ", "Rs. ")) for value in formatted_values)
    assert 0.03 <= len(missing_collateral) / len(records) <= 0.07


def test_generate_applicants_clean_mode_has_no_quality_flags() -> None:
    records = generate_applicants(100, seed=42)

    assert all("data_quality_flags" not in record for record in records)
    assert all(isinstance(record["annual_revenue"], float) for record in records)
    assert all(record["collateral_coverage_ratio"] is not None for record in records)


def test_generate_applicants_adds_code_switched_credit_notes() -> None:
    records = generate_applicants(250, seed=42)
    repeated_records = generate_applicants(250, seed=42)
    notes = [record["credit_officer_notes"] for record in records]

    assert all(isinstance(note, str) and note for note in notes)
    assert notes == [record["credit_officer_notes"] for record in repeated_records]
    assert any(" ka " in f" {note.lower()} " or " ki " in f" {note.lower()} " for note in notes)
    assert any(term in note.lower() for note in notes for term in ("cash flow", "bank", "repayment"))
    assert len(set(notes)) >= 3


def test_generate_final_dataset_writes_versioned_data_and_metadata(tmp_path) -> None:
    csv_path, metadata_path = generate_final_dataset(tmp_path, row_count=10_000, seed=42)

    with csv_path.open(encoding="utf-8") as dataset_file:
        row_count = sum(1 for _ in dataset_file) - 1
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert row_count == 10_000
    assert metadata["dataset_name"] == "creditsense_final_portfolio"
    assert metadata["version"] == "1.0.0"
    assert metadata["row_count"] == 10_000
    assert metadata["messy_mode"] is True
    assert "credit_officer_notes" in metadata["schema"]
    assert "collateral_coverage_ratio" in metadata["schema"]


def test_sanity_check_reports_sector_distributions_and_quality(tmp_path) -> None:
    dataset_path, _ = generate_final_dataset(tmp_path, row_count=10_000, seed=42)
    report = run_sanity_check(dataset_path, tmp_path / "sanity.json")

    assert report["all_checks_pass"] is True
    assert report["row_count"] == 10_000
    assert set(report["sector_counts"]) == {
        "agriculture",
        "food",
        "manufacturing",
        "services",
        "technology",
        "textile",
        "trading",
    }
    assert "annual_revenue" in report["numeric_summary_by_sector"]["textile"]
