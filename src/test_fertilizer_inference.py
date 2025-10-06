#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import joblib

# Robust target detection
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
    raise ValueError('Target column not found')

# Flexible dataset resolution supporting execution from repo root or notebooks/
def find_data_file():
    cwd = Path.cwd()
    candidate_roots = [cwd / 'data', cwd.parent / 'data', cwd.parent.parent / 'data']
    candidate_names = [
        'fertilizer.csv',
        'Fertilizer Prediction.csv',
        'fertlizer_perdiction.csv',
        'fertilizer_prediction.csv',
    ]
    for root in candidate_roots:
        for name in candidate_names:
            for sub in [Path('.'), Path('raw')]:
                p = root / sub / name
                if p.exists():
                    return p
    for root in candidate_roots:
        if root.exists():
            for p in root.rglob('*.csv'):
                n = p.name.lower()
                if 'fert' in n:
                    return p
    raise FileNotFoundError('Could not find fertilizer CSV under data/ or data/raw/.')

DATA_PATH = find_data_file()
# Resolve models when run from repo root or notebooks/
MODELS_DIR = (Path.cwd() if (Path.cwd() / 'models').exists() else Path.cwd().parent) / 'models'

def main():
    model = joblib.load(MODELS_DIR / 'fert_model.joblib')
    le = joblib.load(MODELS_DIR / 'fert_labelencoder.joblib')
    df = pd.read_csv(DATA_PATH)
    y_col = find_target(df)
    X = df.drop(columns=[y_col])
    sample = X.iloc[[0]].copy()
    pred_enc = model.predict(sample)
    pred = le.inverse_transform(pred_enc)
    print('Using DATA_PATH:', DATA_PATH.resolve())
    print('Using MODELS_DIR:', MODELS_DIR.resolve())
    print('Sample input:')
    print(sample.to_dict(orient='records')[0])
    print(f'Predicted fertilizer: {pred[0]}')

if __name__ == '__main__':
    main()
