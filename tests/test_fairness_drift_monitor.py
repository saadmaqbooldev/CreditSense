import pandas as pd

from scripts.fairness_drift_monitor import run_monitoring


def test_monitoring_report_contains_fairness_and_clear_drift(tmp_path) -> None:
    dataset = tmp_path / "reference.csv"
    source = pd.read_csv("data/raw/creditsense_final_portfolio_v1.csv")
    source.to_csv(dataset, index=False)

    report = run_monitoring(dataset, dataset, tmp_path / "monitoring.json")

    assert set(report["fairness"]["by_sector"]) == {
        "agriculture",
        "food",
        "manufacturing",
        "services",
        "technology",
        "textile",
        "trading",
    }
    assert all("tpr_recall" in values for values in report["fairness"]["by_sector"].values())
    assert report["drift"]["alerts"] == []
    assert report["all_drift_alerts_clear"] is True
