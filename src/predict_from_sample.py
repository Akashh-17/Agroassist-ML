"""Inference helper script for Crop Recommendation model.

Usage:
    python -m src.predict_from_sample

Edit the `sample_input` dict in __main__ or import predict() in other modules.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import joblib
from typing import Dict

from src.data_utils import NUMERIC_FEATURES

MODEL_PATH = Path("models/crop_model.joblib")
SCALER_PATH = Path("models/crop_scaler.joblib")
IMPUTER_PATH = Path("models/crop_imputer.joblib")
LABEL_ENCODER_PATH = Path("models/crop_labelencoder.joblib")


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found. Train the model first (run src/train_crop.py)"
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, scaler, imputer, label_encoder


def predict(sample: Dict[str, float]) -> str:
    model, scaler, imputer, label_encoder = load_artifacts()
    # Build DataFrame in the correct feature order
    df = pd.DataFrame([[sample[f] for f in NUMERIC_FEATURES]], columns=NUMERIC_FEATURES)
    X_imp = imputer.transform(df)
    X_scaled = scaler.transform(X_imp)
    pred_idx = model.predict(X_scaled)[0]
    label = label_encoder.inverse_transform([pred_idx])[0]
    return label


if __name__ == "__main__":
    sample_input = {
        "N": 90,
        "P": 40,
        "K": 40,
        "temperature": 24.5,
        "humidity": 80.0,
        "ph": 6.5,
        "rainfall": 200.0,
    }
    result = predict(sample_input)
    print("Sample prediction:", result)
