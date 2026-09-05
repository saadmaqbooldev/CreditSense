"""Dataset loading and validation utilities."""

import pandas as pd
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Load and validate the synthetic dataset."""
    
    REQUIRED_COLUMNS = [
        "applicant_id", "sector", "revenue", "leverage", "growth_rate",
        "margin", "years_in_business", "late_payment_count",
        "collateral_ratio", "default_probability", "requested_credit_limit"
    ]
    
    VALID_SECTORS = [
        "textile", "trading", "food", "manufacturing",
        "services", "agriculture", "technology"
    ]
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.metadata_dir = self.data_dir / "metadata"
        
        # Create directories if they don't exist
        for d in [self.raw_dir, self.processed_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def load_raw(self, filename: str) -> pd.DataFrame:
        """Load the raw synthetic dataset."""
        filepath = self.raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        # Auto-detect file format
        if filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(filepath)
        elif filename.endswith(".json"):
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        logger.info(f"Loaded {len(df)} records from {filepath}")
        return df
    
    def validate(self, df: pd.DataFrame) -> dict:
        """Validate the dataset structure and data quality."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # 1. Check required columns
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            results["valid"] = False
            results["errors"].append(f"Missing columns: {missing}")
        
        # 2. Record count
        results["stats"]["total_records"] = len(df)
        results["stats"]["total_columns"] = len(df.columns)
        
        # 3. Duplicate check
        duplicates = df.duplicated().sum()
        results["stats"]["duplicates"] = duplicates
        if duplicates > 0:
            results["warnings"].append(f"Found {duplicates} duplicate rows")
        
        # 4. Missing values
        missing_counts = df.isnull().sum()
        results["stats"]["missing_by_column"] = missing_counts.to_dict()
        total_missing = missing_counts.sum()
        results["stats"]["total_missing"] = total_missing
        if total_missing > 0:
            results["warnings"].append(f"Total missing values: {total_missing}")
        
        # 5. Sector validation
        if "sector" in df.columns:
            invalid_sectors = set(df["sector"].unique()) - set(self.VALID_SECTORS)
            if invalid_sectors:
                results["warnings"].append(f"Invalid sectors found: {invalid_sectors}")
            results["stats"]["sector_distribution"] = df["sector"].value_counts().to_dict()
        
        # 6. Data type inference
        results["stats"]["dtypes"] = df.dtypes.astype(str).to_dict()
        
        # 7. Numeric range checks
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            results["stats"][f"{col}_min"] = float(df[col].min())
            results["stats"][f"{col}_max"] = float(df[col].max())
            results["stats"][f"{col}_mean"] = float(df[col].mean())
        
        # 8. Class distribution (target variable)
        if "default_probability" in df.columns:
            # Create binary default class (threshold at 0.5)
            df["default_class"] = (df["default_probability"] >= 0.5).astype(int)
            results["stats"]["class_distribution"] = df["default_class"].value_counts().to_dict()
        
        return results
    
    def save_metadata(self, df: pd.DataFrame, results: dict, filename: str = "dataset_metadata.json"):
        """Save dataset metadata."""
        metadata = {
            "dataset_name": filename,
            "created_at": datetime.now().isoformat(),
            "total_records": results["stats"]["total_records"],
            "total_columns": results["stats"]["total_columns"],
            "columns": list(df.columns),
            "sector_distribution": results["stats"].get("sector_distribution", {}),
            "class_distribution": results["stats"].get("class_distribution", {}),
            "missing_count": results["stats"].get("total_missing", 0),
            "duplicate_count": results["stats"].get("duplicates", 0),
            "dtypes": results["stats"].get("dtypes", {}),
            "validation_errors": results.get("errors", []),
            "validation_warnings": results.get("warnings", [])
        }
        
        filepath = self.metadata_dir / "dataset_metadata.json"
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {filepath}")
        return metadata