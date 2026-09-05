"""
Threshold Tuning for XGBoost Model - Find Optimal Threshold for 54% Recall
Run this after training to find the best threshold.
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml_pipeline import engineer_features, FEATURE_COLUMNS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
ARTIFACT_DIR = BASE_DIR / "app" / "ml_artifacts"

def load_model():
    """Load the trained XGBoost model."""
    model_path = ARTIFACT_DIR / "xgboost_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"✅ Model loaded from: {model_path}")
    return model


def load_data(filepath: Path) -> pd.DataFrame:
    """Load Dataset 2."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"✅ Loaded {len(df)} records from {filepath}")
    return df


def prepare_test_data(df: pd.DataFrame):
    """Prepare test data using same logic as training script."""
    
    data = df.copy()
    
    # Column mapping
    column_mapping = {
        'debt_to_equity_ratio': 'debt_to_equity',
        'revenue_growth_yoy_pct': 'revenue_growth_yoy',
        'collateral_coverage_ratio': 'collateral_coverage',
        'annual_bank_turnover_pkr': 'revenue',
        'years_in_business': 'years_in_business',
        'late_payment_count_12m': 'late_payment_count',
        'current_ratio': 'current_ratio',
        'bank_statement_volatility': 'bank_statement_volatility',
        'existing_loan_exposure_pkr': 'existing_loan_exposure',
        'default_probability_12m': 'default_probability_12m',
        'recommend_credit_limit_pkr': 'recommended_credit_limit',
    }
    
    data = data.rename(columns=column_mapping)
    
    # Create leverage
    if 'debt_to_equity' in data.columns:
        data['leverage'] = data['debt_to_equity'] / (1 + data['debt_to_equity'])
    else:
        data['leverage'] = 0.5
    
    # Create margin
    if 'revenue' in data.columns:
        data['annual_expenses'] = data['revenue'] * (0.7 + (data['leverage'] * 0.2))
        data['margin'] = (data['revenue'] - data['annual_expenses']) / data['revenue']
        data['margin'] = data['margin'].clip(0, 1)
    else:
        data['margin'] = 0.2
    
    # Interaction features
    if 'leverage' in data.columns and 'revenue_growth_yoy' in data.columns:
        data['leverage_x_growth'] = data['leverage'] * data['revenue_growth_yoy']
    else:
        data['leverage_x_growth'] = 0
    
    if 'debt_to_equity' in data.columns and 'collateral_coverage' in data.columns:
        data['debt_x_collateral'] = data['debt_to_equity'] * data['collateral_coverage']
    else:
        data['debt_x_collateral'] = 0
    
    if 'late_payment_count' in data.columns and 'years_in_business' in data.columns:
        data['payment_history'] = data['late_payment_count'] / (data['years_in_business'] + 1)
    else:
        data['payment_history'] = 0
    
    # Debt service ratio
    if 'debt_service_ratio' not in data.columns:
        if 'debt_to_equity' in data.columns and 'revenue' in data.columns:
            data['debt_service_ratio'] = data['debt_to_equity'] / (data['revenue'] / 1000000 + 1)
        else:
            data['debt_service_ratio'] = 0.3
    
    # Liquidity ratio
    if 'liquidity_ratio' not in data.columns:
        if 'current_ratio' in data.columns:
            data['liquidity_ratio'] = data['current_ratio'] * 0.8
        else:
            data['liquidity_ratio'] = 1.0
    
    # Sector risk score
    if 'sector_risk_score' not in data.columns:
        if 'sector_risk_weight' in df.columns:
            data['sector_risk_score'] = df['sector_risk_weight']
        else:
            data['sector_risk_score'] = 3
    
    # Features
    features = [
        'leverage', 'debt_to_equity', 'revenue_growth_yoy', 
        'years_in_business', 'late_payment_count', 'collateral_coverage',
        'debt_service_ratio', 'liquidity_ratio', 'bank_statement_volatility',
        'sector_risk_score', 'current_ratio', 'leverage_x_growth',
        'debt_x_collateral', 'payment_history'
    ]
    
    X = data[features]
    
    # Target
    if 'default_probability_12m' in df.columns:
        y = (df['default_probability_12m'] >= 0.5).astype(int)
    elif 'default_12m' in df.columns:
        y = df['default_12m']
    else:
        raise ValueError("No target column found")
    
    # Fill NaN
    X = X.fillna(0)
    
    return X, y


def find_optimal_threshold(y_test, y_pred_proba):
    """
    Find optimal threshold for different scenarios.
    Returns: threshold_recall_54, threshold_f1_best, threshold_precision_recall_balance
    """
    
    # 1. Find threshold for 54% recall
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Remove the last element (thresholds are one less than precision/recall)
    thresholds = thresholds[:-1] if len(thresholds) > 0 else np.array([0.5])
    precision_vals = precision_vals[:-1]
    recall_vals = recall_vals[:-1]
    
    # Find threshold closest to 54% recall
    threshold_recall_54 = 0.50
    for i, recall_val in enumerate(recall_vals):
        if recall_val <= 0.55 and recall_val >= 0.53:
            threshold_recall_54 = thresholds[i]
            break
    
    # 2. Find best threshold for F1 score
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    threshold_f1_best = thresholds[best_f1_idx]
    
    # 3. Find threshold where precision = recall (balanced)
    balanced_idx = np.argmin(np.abs(precision_vals - recall_vals))
    threshold_balanced = thresholds[balanced_idx]
    
    return {
        'threshold_recall_54': threshold_recall_54,
        'threshold_f1_best': threshold_f1_best,
        'threshold_balanced': threshold_balanced,
        'precision_vals': precision_vals,
        'recall_vals': recall_vals,
        'thresholds': thresholds,
        'f1_scores': f1_scores,
        'best_f1_score': f1_scores[best_f1_idx]
    }


def evaluate_at_threshold(y_test, y_pred_proba, threshold, name):
    """Evaluate model performance at a specific threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 {name}")
    logger.info(f"{'='*50}")
    logger.info(f"  Threshold:      {threshold:.4f}")
    logger.info(f"  Accuracy:       {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:      {metrics['precision']:.4f}")
    logger.info(f"  Recall:         {metrics['recall']:.4f}  {('✅' if 0.52 <= metrics['recall'] <= 0.56 else '')}")
    logger.info(f"  F1 Score:       {metrics['f1_score']:.4f}")
    logger.info(f"  AUC-ROC:        {metrics['auc_roc']:.4f}")
    logger.info(f"  Confusion Matrix: {metrics['confusion_matrix']}")
    
    return metrics


def main():
    """Main threshold tuning."""
    logger.info("="*60)
    logger.info("🎯 THRESHOLD TUNING FOR XGBOOST MODEL")
    logger.info("="*60)
    
    # 1. Load model
    model = load_model()
    
    # 2. Load Dataset 2
    possible_paths = [
        Path("creditsense_synthetic_portfolio.csv"),
        DATA_DIR / "creditsense_synthetic_portfolio.csv",
    ]
    
    dataset_path = None
    for path in possible_paths:
        if path.exists():
            dataset_path = path
            break
    
    if dataset_path is None:
        logger.error("❌ Dataset not found!")
        return
    
    df = load_data(dataset_path)
    
    # 3. Prepare test data
    X, y = prepare_test_data(df)
    
    # Split into train/test (use same random_state as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    logger.info(f"Test set: {len(X_test)} records")
    logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    # 4. Get probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 5. Find optimal thresholds
    results = find_optimal_threshold(y_test, y_pred_proba)
    
    logger.info("\n" + "="*50)
    logger.info("🎯 OPTIMAL THRESHOLDS FOUND")
    logger.info("="*50)
    logger.info(f"  For 54% Recall:    {results['threshold_recall_54']:.4f}")
    logger.info(f"  Best F1 Score:      {results['threshold_f1_best']:.4f} (F1={results['best_f1_score']:.4f})")
    logger.info(f"  Precision=Recall:   {results['threshold_balanced']:.4f}")
    
    # 6. Evaluate each threshold
    metrics_54 = evaluate_at_threshold(y_test, y_pred_proba, results['threshold_recall_54'], "54% RECALL TARGET")
    metrics_f1 = evaluate_at_threshold(y_test, y_pred_proba, results['threshold_f1_best'], "BEST F1 SCORE")
    metrics_balanced = evaluate_at_threshold(y_test, y_pred_proba, results['threshold_balanced'], "PRECISION = RECALL")
    
    # 7. Save results
    output_path = ARTIFACT_DIR / "threshold_analysis.json"
    threshold_results = {
        'thresholds': {
            'recall_54': results['threshold_recall_54'],
            'best_f1': results['threshold_f1_best'],
            'balanced': results['threshold_balanced']
        },
        'metrics': {
            'recall_54': metrics_54,
            'best_f1': metrics_f1,
            'balanced': metrics_balanced
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(threshold_results, f, indent=2)
    logger.info(f"\n✅ Threshold analysis saved to: {output_path}")
    
    # 8. Recommendation
    logger.info("\n" + "="*50)
    logger.info("📋 RECOMMENDATION")
    logger.info("="*50)
    
    # Which threshold to recommend?
    # Target is 54% recall, so we recommend that threshold
    recommended = results['threshold_recall_54']
    
    logger.info(f"  🎯 Use threshold: {recommended:.4f}")
    logger.info(f"     Recall:  {metrics_54['recall']:.4f} (target: ~0.54)")
    logger.info(f"     Precision: {metrics_54['precision']:.4f}")
    logger.info(f"     F1 Score: {metrics_54['f1_score']:.4f}")
    logger.info("\n  To use this threshold in production:")
    logger.info(f"  y_pred = (model.predict_proba(X)[:, 1] >= {recommended:.4f}).astype(int)")
    
    logger.info("\n" + "="*60)
    logger.info("✅ THRESHOLD TUNING COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    main()