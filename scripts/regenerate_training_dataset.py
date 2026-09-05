"""Regenerate the synthetic portfolio target from observable credit-risk drivers."""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_PATH = BASE_DIR / "creditsense_synthetic_portfolio.csv"
OUTPUT_PATHS = [
    SOURCE_PATH,
    BASE_DIR / "data" / "raw" / "creditsense_synthetic_portfolio.csv",
]
DEFAULT_RATE = 0.10
NOISE_SCALE = 6.5
SEED = 42


def z_score(series: pd.Series) -> pd.Series:
    values = series.fillna(series.median())
    return (values - values.mean()) / (values.std() + 1e-9)


def build_default_target(data: pd.DataFrame) -> pd.Series:
    """Build a noisy, rankable default label from observable risk drivers."""
    risk_score = (
        1.8 * z_score(data["risk_index_score"])
        + 1.2 * z_score(data["bank_statement_volatility"])
        + 1.0 * z_score(data["late_payment_count_12m"])
        + 0.8 * z_score(data["working_capital_cycle_days"])
        + 0.7 * z_score(data["multi_banking_behavior"])
        + 0.6 * z_score(data["debt_to_equity_ratio"])
        - 0.8 * z_score(data["current_ratio"])
        - 0.7 * z_score(data["collateral_coverage_ratio"])
        - 0.6 * z_score(data["declared_income_ratio"])
        - 0.5 * z_score(data["revenue_growth_yoy_pct"])
    )

    rng = np.random.default_rng(SEED)
    noisy_score = risk_score + rng.normal(0, NOISE_SCALE, len(data))
    cutoff = noisy_score.quantile(1 - DEFAULT_RATE)
    return (noisy_score >= cutoff).astype(int)


def main() -> None:
    data = pd.read_csv(SOURCE_PATH)
    data["default_probability_12m"] = build_default_target(data)
    data["target_generation"] = f"risk_drivers_v3_noise_{NOISE_SCALE}"

    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"Wrote {len(data)} rows to {output_path}")

    print(f"Default rate: {data['default_probability_12m'].mean():.2%}")


if __name__ == "__main__":
    main()