#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Flexible dataset resolution supporting execution from repo root or notebooks/
def find_data_file():
    cwd = Path.cwd()
    candidate_roots = [cwd / 'data', cwd.parent / 'data', cwd.parent.parent / 'data']
    candidate_names = [
        'fertilizer.csv',
        'Fertilizer Prediction.csv',
        # tolerate common misspellings
        'fertlizer_perdiction.csv',
        'fertilizer_prediction.csv',
    ]
    # Direct candidates in data/ and data/raw/
    for root in candidate_roots:
        for name in candidate_names:
            for sub in [Path('.'), Path('raw')]:
                p = root / sub / name
                if p.exists():
                    return p
    # Fallback: search any CSV with 'fert' in name under data trees
    for root in candidate_roots:
        if root.exists():
            for p in root.rglob('*.csv'):
                n = p.name.lower()
                if 'fert' in n:
                    return p
    raise FileNotFoundError('Could not find fertilizer CSV under data/ or data/raw/.')

DATA_PATH = find_data_file()
# Ensure models/ resolves correctly when run from repo root or notebooks/
MODELS_DIR = (Path.cwd() if (Path.cwd() / 'models').exists() else Path.cwd().parent) / 'models'

# Robust target detection that tolerates spaces/underscores/case
_DEF_TARGET_ALIASES = [
    'fertilizer_name', 'fertilizername', 'fertilizer',
    'fertilizerlabel', 'label', 'target'
]

def _norm(s):
    return str(s).strip().lower().replace('_', '').replace(' ', '')

def find_target(df):
    normalized_cols = { _norm(c): c for c in df.columns }
    for alias in _DEF_TARGET_ALIASES:
        if alias in normalized_cols:
            return normalized_cols[alias]
    fert_like = [c for c in df.columns if 'fertilizer' in c.lower()]
    if len(fert_like) == 1:
        return fert_like[0]
    raise ValueError('Target column not found in dataset')

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    y_col = find_target(df)
    X = df.drop(columns=[y_col])
    y = df[y_col].astype(str)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    print(f'Using DATA_PATH: {DATA_PATH.resolve()}')
    print(f'Using target: {y_col}')
    print(f'Numeric: {num_cols}')
    print(f'Categorical: {cat_cols}')
    # Label encode target and save
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, MODELS_DIR / 'fert_labelencoder.joblib')
    # Preprocess features (guard empty column lists)
    transformers = []
    if len(num_cols) > 0:
        transformers.append(('num', StandardScaler(), num_cols))
    if len(cat_cols) > 0:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols))
    if not transformers:
        raise ValueError('No features found to train on (no numeric or categorical columns).')
    preprocessor = ColumnTransformer(transformers)
    # Model (class_weight helps imbalance)
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced')
    pipe = Pipeline(steps=[('preprocess', preprocessor), ('clf', clf)])
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    # Train
    pipe.fit(X_train, y_train)
    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')
    print('Classification report:')
    print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))
    # Save model pipeline and scaler artifact
    joblib.dump(pipe, MODELS_DIR / 'fert_model.joblib')
    # Save scaler if numeric branch exists
    try:
        scaler = pipe.named_steps['preprocess'].named_transformers_.get('num')
    except Exception:
        scaler = None
    if scaler is not None:
        joblib.dump(scaler, MODELS_DIR / 'fert_scaler.joblib')
    print('Saved models/fert_model.joblib, models/fert_scaler.joblib (if any), models/fert_labelencoder.joblib')

if __name__ == '__main__':
    sys.exit(main())
