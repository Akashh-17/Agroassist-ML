from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, List
import joblib
import numpy as np
import pandas as pd

# Lazy-loaded globals
_MODEL = None
_SCALER = None
_LABELER = None
_FEATURE_COLUMNS: Optional[List[str]] = None

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)


def load_artifacts(models_dir: Optional[str | Path] = None):
    global _MODEL, _SCALER, _LABELER, _FEATURE_COLUMNS
    mdir = Path(models_dir) if models_dir else MODELS_DIR
    model_path = mdir / "crop_best_model.joblib"
    if not model_path.exists():
        # fallback to baseline
        model_path = mdir / "crop_model.joblib"
    scaler_path = mdir / "crop_scaler.joblib"
    labeler_path = mdir / "crop_labelencoder.joblib"

    _MODEL = _load_artifact(model_path)
    _SCALER = _load_artifact(scaler_path)
    _LABELER = _load_artifact(labeler_path)

    # Determine feature columns from scaler if available
    if hasattr(_SCALER, "feature_names_in_"):
        _FEATURE_COLUMNS = list(_SCALER.feature_names_in_)
    else:
        # default expected order for this project
        _FEATURE_COLUMNS = [
            "N", "P", "K", "temperature", "humidity", "ph", "rainfall"
        ]


def preprocess(input_dict: Dict[str, float]) -> np.ndarray:
    if _SCALER is None or _FEATURE_COLUMNS is None:
        load_artifacts()
    # Build a one-row DataFrame in the expected column order
    row = {col: pd.to_numeric(input_dict.get(col, np.nan), errors="coerce") for col in _FEATURE_COLUMNS}
    X_df = pd.DataFrame([row], columns=_FEATURE_COLUMNS)
    # Scale consistently:
    # - If scaler knows feature names, give it a DataFrame
    # - If scaler was fitted without names, pass numpy to avoid warnings, then wrap back to DataFrame
    if hasattr(_SCALER, "transform"):
        if hasattr(_SCALER, "feature_names_in_"):
            X_scaled = _SCALER.transform(X_df)
        else:
            X_scaled = _SCALER.transform(X_df.to_numpy())
    else:
        X_scaled = X_df.to_numpy()
    # Return as DataFrame with columns so downstream models fitted with feature names don't warn
    X_scaled_df = pd.DataFrame(X_scaled, columns=_FEATURE_COLUMNS)
    return X_scaled_df


def predict_crop(input_dict: Dict[str, float], models_dir: Optional[str | Path] = None) -> str:
    if any(obj is None for obj in (_MODEL, _SCALER, _LABELER)):
        load_artifacts(models_dir)
    X = preprocess(input_dict)
    y_pred = _MODEL.predict(X)
    # If model outputs encoded labels, decode with label encoder
    if hasattr(_LABELER, "inverse_transform"):
        label = _LABELER.inverse_transform(y_pred)[0]
    else:
        label = str(y_pred[0])
    return label


__all__ = ["load_artifacts", "preprocess", "predict_crop"]
