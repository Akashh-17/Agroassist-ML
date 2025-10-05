"""Utility functions for Crop Recommendation preprocessing.

This module centralizes data loading, preprocessing (imputation + scaling),
label encoding, and artifact persistence so that both training and
inference scripts remain clean and consistent.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Any, List
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Core configuration
NUMERIC_FEATURES: List[str] = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
TARGET_COLUMN: str = "label"

# Potential dataset filenames (the repo currently contains Crop_recommendation.csv)
DATA_CANDIDATES = [
    Path("data/crop.csv"),
    Path("data/Crop_recommendation.csv"),
]


def resolve_dataset_path() -> Path:
    """Return the first existing dataset path among candidates.

    Raises
    ------
    FileNotFoundError
        If none of the candidate files exist.
    """
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"None of the expected dataset files found: {[str(p) for p in DATA_CANDIDATES]}"
    )


def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load dataset into a DataFrame.

    Parameters
    ----------
    csv_path : optional explicit path. If None, auto-resolve.
    """
    if csv_path is None:
        csv_path = resolve_dataset_path()
    return pd.read_csv(csv_path)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the DataFrame into feature matrix X and target vector y."""
    X = df[NUMERIC_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_feature_pipeline() -> Tuple[SimpleImputer, StandardScaler]:
    """Instantiate the preprocessing components.

    Returns
    -------
    (imputer, scaler)
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    return imputer, scaler


def fit_transform_features(
    X: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler
) -> np.ndarray:
    """Fit imputer & scaler on X then return the transformed array."""
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled


def transform_features(
    X: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler
) -> np.ndarray:
    """Apply already-fitted imputer & scaler to new data."""
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    return X_scaled


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """Fit a LabelEncoder and transform y.

    Returns
    -------
    y_encoded, fitted_label_encoder
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def save_artifact(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str | Path) -> Any:
    return joblib.load(path)
