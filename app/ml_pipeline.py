"""ML Pipeline: Feature engineering and scoring."""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# SECTION 1: FEATURE DEFINITIONS
# ============================================================

# Define all feature columns used by the model
FEATURE_COLUMNS = [
    'leverage',
    'debt_to_equity',
    'revenue_growth_yoy',
    'years_in_business',
    'late_payment_count',
    'collateral_coverage',
    'debt_service_ratio',
    'liquidity_ratio',
    'bank_statement_volatility',
    'sector_risk_score',
    'current_ratio',
    'leverage_x_growth',
    'debt_x_collateral',
    'payment_history'
]

# Define sector mapping for one-hot encoding
SECTOR_MAPPING = {
    'textile': 'sector_textile',
    'trading': 'sector_trading',
    'food': 'sector_food',
    'manufacturing': 'sector_manufacturing',
    'services': 'sector_services',
    'agriculture': 'sector_agriculture',
    'technology': 'sector_technology'
}


# ============================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw applicant data.
    
    This function takes raw input data and creates all features needed
    for the XGBoost model.
    
    Args:
        df: DataFrame with raw applicant data
        
    Returns:
        DataFrame with engineered features (includes target if present)
    """
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # 1. Leverage (ensure in [0, 1] range)
    if 'leverage' in data.columns:
        data['leverage'] = data['leverage'].clip(0, 1)
    else:
        # Try debt_to_equity as fallback
        if 'debt_to_equity' in data.columns:
            data['leverage'] = data['debt_to_equity'] / (1 + data['debt_to_equity'])
        else:
            data['leverage'] = 0.5  # Default
    
    # 2. Debt-to-equity ratio
    if 'debt_to_equity' not in data.columns:
        if 'leverage' in data.columns:
            data['debt_to_equity'] = data['leverage'] / (1 - data['leverage'].clip(0.01, 0.99))
        else:
            data['debt_to_equity'] = 1.0
    
    # 3. Revenue growth (YoY)
    if 'revenue_growth_yoy' not in data.columns:
        if 'growth_rate' in data.columns:
            data['revenue_growth_yoy'] = data['growth_rate']
        else:
            data['revenue_growth_yoy'] = 0.05
    
    # 4. Margin
    if 'margin' not in data.columns:
        data['margin'] = 0.10
    
    # 5. Years in business
    if 'years_in_business' not in data.columns:
        data['years_in_business'] = 5
    
    # 6. Late payment count
    if 'late_payment_count' not in data.columns:
        data['late_payment_count'] = 0
    
    # 7. Collateral coverage
    if 'collateral_coverage' not in data.columns:
        if 'collateral_ratio' in data.columns:
            data['collateral_coverage'] = data['collateral_ratio']
        else:
            data['collateral_coverage'] = 0.8
    
    # 8. One-hot encode sector
    if 'sector' in data.columns:
        for sector in SECTOR_MAPPING.keys():
            col_name = SECTOR_MAPPING[sector]
            data[col_name] = (data['sector'] == sector).astype(int)
    
    # Ensure all FEATURE_COLUMNS exist
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            if col.startswith('sector_'):
                data[col] = 0
            elif col == 'debt_to_equity':
                data[col] = 1.0
            elif col == 'revenue_growth_yoy':
                data[col] = 0.05
            elif col == 'collateral_coverage':
                data[col] = 0.8
            else:
                data[col] = 0.0
    
    return data


# ============================================================
# SECTION 3: MODEL LOADING AND SCORING
# ============================================================

def build_feature_vector(applicant_data: dict) -> dict:
    """Build the stable scalar feature contract used by the API baseline."""
    debt_to_equity = float(applicant_data.get("debt_to_equity", 0.0) or 0.0)
    revenue_growth = float(applicant_data.get("revenue_growth_yoy", 0.0) or 0.0)
    operating_margin = float(applicant_data.get("operating_margin", 0.0) or 0.0)
    years_in_business = float(applicant_data.get("years_in_business", 0.0) or 0.0)
    late_payment_count = float(applicant_data.get("late_payment_count", 0.0) or 0.0)
    collateral = float(applicant_data.get("collateral_coverage_ratio", 0.0) or 0.0)

    return {
        "debt_to_equity": debt_to_equity,
        "revenue_growth_yoy": revenue_growth,
        "operating_margin": operating_margin,
        "years_in_business": years_in_business,
        "late_payment_count": late_payment_count,
        "collateral_coverage_ratio": collateral,
        "debt_to_equity_sq": debt_to_equity ** 2,
        "leverage_risk": debt_to_equity / (1.0 + debt_to_equity),
    }


def score_applicant(applicant_data: dict) -> float:
    """Return a bounded default-risk probability for the API and agents."""
    features = build_feature_vector(applicant_data)
    debt_risk = min(features["debt_to_equity"] / 3.0, 1.0)
    payment_risk = min(features["late_payment_count"] / 10.0, 1.0)
    collateral_risk = 1.0 - min(features["collateral_coverage_ratio"], 1.0)
    growth_risk = min(max(0.2 - features["revenue_growth_yoy"], 0.0), 0.5)
    margin_risk = min(max(0.2 - features["operating_margin"], 0.0), 0.5)
    experience_risk = 1.0 - min(features["years_in_business"] / 10.0, 1.0)

    probability = (
        0.10
        + 0.18 * debt_risk
        + 0.14 * payment_risk
        + 0.12 * collateral_risk
        + 0.10 * growth_risk
        + 0.10 * margin_risk
        + 0.06 * experience_risk
    )
    return float(max(0.0, min(1.0, probability)))

def load_model(model_path: str = None):
    """
    Load the trained XGBoost model from artifacts.
    
    Args:
        model_path: Optional path to model file. If None, uses default path.
    
    Returns:
        Loaded XGBoost model
    """
    if model_path is None:
        base_dir = Path(__file__).parent
        model_path = base_dir / "ml_artifacts" / "xgboost_model.pkl"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = joblib.load(model_path)
    return model


def load_model_metadata():
    """Load model metadata."""
    base_dir = Path(__file__).parent
    metadata_path = base_dir / "ml_artifacts" / "model_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def score_default_risk(applicant_data: dict) -> dict:
    """
    Score default risk using the trained XGBoost model.
    Falls back to the baseline logistic model if XGBoost model is not available.
    """
    try:
        # Try to load the trained XGBoost model
        model = load_model()
        metadata = load_model_metadata()
        
        # Prepare features
        df = pd.DataFrame([applicant_data])
        df_engineered = engineer_features(df)
        
        # Get feature matrix
        X = df_engineered[FEATURE_COLUMNS]
        
        # Predict
        default_prob = model.predict_proba(X)[0, 1]
        
        return {
            'default_probability': float(default_prob),
            'model_version': metadata.get('version', 'xgboost-1.0.0'),
            'model_type': 'xgboost'
        }
        
    except (FileNotFoundError, Exception) as e:
        # Fallback to baseline linear model
        print(f"Using fallback model: {e}")
        return baseline_score_default_risk(applicant_data)


def baseline_score_default_risk(applicant_data: dict) -> dict:
    """
    Baseline logistic scoring function (original implementation).
    """
    leverage = applicant_data.get('leverage', 0.5)
    growth = applicant_data.get('revenue_growth_yoy', 0.0)
    margin = applicant_data.get('margin', 0.1)
    years = applicant_data.get('years_in_business', 5)
    late_payments = applicant_data.get('late_payment_count', 0)
    collateral = applicant_data.get('collateral_coverage', 1.0)
    
    score = (
        0.30 * leverage +
        -0.15 * growth +
        -0.20 * margin +
        -0.05 * min(years / 10, 1) +
        0.15 * min(late_payments / 10, 1) +
        -0.10 * min(collateral, 1)
    )
    
    default_prob = max(0, min(1, score))
    
    return {
        'default_probability': default_prob,
        'model_version': 'baseline-v1',
        'model_type': 'logistic'
    }


def predict_risk_from_dict(applicant_data: dict) -> dict:
    """
    Predict risk from a dictionary of applicant data.
    This is the function called by the /predict/credit-risk endpoint.
    """
    result = score_default_risk(applicant_data)
    
    prob = result['default_probability']
    if prob < 0.3:
        risk_band = "low"
    elif prob < 0.6:
        risk_band = "medium"
    else:
        risk_band = "high"
    
    result['risk_band'] = risk_band
    
    return result