"""Training script for Crop Recommendation baseline model.

Steps:
1. Load dataset (auto-resolve path among expected filenames)
2. Split features & target
3. Encode labels
4. Train/test split
5. Fit preprocessing (imputer + scaler) on train, transform test
6. Train RandomForestClassifier
7. Evaluate (accuracy + classification report)
8. Persist model + preprocessing artifacts
"""

from __future__ import annotations
from pathlib import Path
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

from src.data_utils import (
    load_data,
    split_features_target,
    build_feature_pipeline,
    fit_transform_features,
    transform_features,
    encode_labels,
    save_artifact,
    NUMERIC_FEATURES,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "crop_model.joblib"
SCALER_PATH = MODELS_DIR / "crop_scaler.joblib"
IMPUTER_PATH = MODELS_DIR / "crop_imputer.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "crop_labelencoder.joblib"


def main():
    warnings.filterwarnings("ignore")

    print("Loading data ...")
    df = load_data()
    print(f"Data shape: {df.shape}")

    print("Splitting features and target ...")
    X, y = split_features_target(df)

    print("Encoding labels ...")
    y_enc, label_encoder = encode_labels(y)

    print("Train/test split ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )

    print("Building preprocessing components ...")
    imputer, scaler = build_feature_pipeline()

    print("Fitting preprocessing on training data ...")
    X_train_proc = fit_transform_features(X_train, imputer, scaler)
    X_test_proc = transform_features(X_test, imputer, scaler)

    print("Training RandomForestClassifier ...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None,  # change to 'balanced' if strong class imbalance
    )
    clf.fit(X_train_proc, y_train)

    print("Evaluating ...")
    y_pred = clf.predict(X_test_proc)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Basic feature importances
    importances = clf.feature_importances_
    fi_pairs = sorted(
        zip(NUMERIC_FEATURES, importances), key=lambda x: x[1], reverse=True
    )
    print("Feature importances:")
    for name, val in fi_pairs:
        print(f"  {name:15s} {val:.4f}")

    print("Saving artifacts ...")
    joblib.dump(clf, MODEL_PATH)
    save_artifact(scaler, SCALER_PATH)
    save_artifact(imputer, IMPUTER_PATH)
    save_artifact(label_encoder, LABEL_ENCODER_PATH)

    print("Done. Artifacts saved:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Scaler: {SCALER_PATH}")
    print(f"  Imputer: {IMPUTER_PATH}")
    print(f"  LabelEncoder: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
