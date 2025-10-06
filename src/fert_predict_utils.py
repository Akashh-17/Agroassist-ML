from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, List
import joblib
import numpy as np
import pandas as pd

_MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'

# Lazy-loaded globals
_MODEL = None
_LABELER = None
_PREPROCESSOR = None  # ColumnTransformer from training
_FEATURE_COLUMNS: Optional[List[str]] = None


def load_artifacts(models_dir: Optional[str | Path] = None, use_best: bool = True):
    global _MODEL, _LABELER, _PREPROCESSOR, _FEATURE_COLUMNS
    mdir = Path(models_dir) if models_dir else _MODELS_DIR
    model_name = 'fert_best_model.joblib' if use_best else 'fert_model.joblib'
    # Fallbacks: per-model artifacts
    candidates = [mdir / model_name]
    for alt in ['rf', 'xgb', 'knn']:
        candidates.append(mdir / f'fert_model_{alt}.joblib')

    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(f"No fertilizer model found under {mdir}")

    _MODEL = joblib.load(model_path)
    # The model pipeline contains the preprocessor as first step
    if hasattr(_MODEL, 'named_steps') and 'preprocess' in _MODEL.named_steps:
        _PREPROCESSOR = _MODEL.named_steps['preprocess']
        # Try to extract numeric feature names if available
        try:
            _FEATURE_COLUMNS = []
            if hasattr(_PREPROCESSOR, 'transformers_'):
                for name, trans, cols in _PREPROCESSOR.transformers_:
                    if isinstance(cols, list):
                        _FEATURE_COLUMNS.extend(cols)
        except Exception:
            _FEATURE_COLUMNS = None
    # Load label encoder
    _LABELER = joblib.load(mdir / 'fert_labelencoder.joblib')


def preprocess(input_dict: Dict[str, object]) -> pd.DataFrame:
    # Build DataFrame with whatever keys user provides; the pipeline will align columns
    X_df = pd.DataFrame([input_dict])
    return X_df


def predict_fertilizer(input_dict: Dict[str, object], models_dir: Optional[str | Path] = None) -> str:
    if any(obj is None for obj in (_MODEL, _LABELER)):
        load_artifacts(models_dir)
    X = preprocess(input_dict)
    pred_enc = _MODEL.predict(X)
    label = _LABELER.inverse_transform(pred_enc)[0]
    return label

__all__ = [
    'load_artifacts',
    'preprocess',
    'predict_fertilizer',
]
