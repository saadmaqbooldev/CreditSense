from app.data_generation import ApplicantSector, generate_applicants


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
