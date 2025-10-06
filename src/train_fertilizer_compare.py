#!/usr/bin/env python3
"""Train and compare multiple models for Fertilizer Recommendation.

Models:
- RandomForestClassifier
- XGBClassifier
- KNeighborsClassifier (as the third baseline)

Outputs:
- figures/fert_model_eval_metrics.csv (+ timestamped copy)
- figures/confusion_fert_<model>.png
- figures/report_fert_<model>.txt
- models/fert_model_<model>.joblib (each)
- models/fert_best_model.joblib + models/fert_best_model_meta.json
- models/fert_labelencoder.joblib (target encoder)
- models/fert_scaler.joblib (only if numeric preprocessor exists)
"""
from __future__ import annotations
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ---------- Robust data and target resolution ----------

def _norm(s: str) -> str:
    return str(s).strip().lower().replace('_', '').replace(' ', '')


def find_data_file() -> Path:
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
    # Fallback search
    for root in candidate_roots:
        if root.exists():
            for p in root.rglob('*.csv'):
                if 'fert' in p.name.lower():
                    return p
    raise FileNotFoundError('Could not find fertilizer CSV under data/ or data/raw/.')


_DEF_TARGET_ALIASES = [
    'fertilizer_name', 'fertilizername', 'fertilizer',
    'fertilizerlabel', 'label', 'target'
]


def find_target(df: pd.DataFrame) -> str:
    normalized_cols = {_norm(c): c for c in df.columns}
    for alias in _DEF_TARGET_ALIASES:
        if alias in normalized_cols:
            return normalized_cols[alias]
    fert_like = [c for c in df.columns if 'fertilizer' in c.lower()]
    if len(fert_like) == 1:
        return fert_like[0]
    raise ValueError('Target column not found in dataset')


# ---------- Training & Evaluation ----------

def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], title: str, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    # Consistent repo-root anchored dirs
    REPO_ROOT = Path(__file__).resolve().parents[1]
    FIG_DIR = REPO_ROOT / 'figures'
    MODELS_DIR = REPO_ROOT / 'models'
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data_path = find_data_file()
    df = pd.read_csv(data_path)

    y_col = find_target(df)
    X = df.drop(columns=[y_col])
    y = df[y_col].astype(str)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, MODELS_DIR / 'fert_labelencoder.joblib')

    # Preprocessor
    transformers = []
    if len(num_cols) > 0:
        transformers.append(('num', StandardScaler(), num_cols))
    if len(cat_cols) > 0:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols))
    if not transformers:
        raise ValueError('No features found to train on (no numeric or categorical columns).')
    preprocessor = ColumnTransformer(transformers)

    # Models
    models: dict[str, object] = {
        'rf': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced'),
        'knn': KNeighborsClassifier(n_neighbors=5),
    }
    if _HAS_XGB:
        models['xgb'] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            objective='multi:softprob',
            eval_metric='mlogloss',
        )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    rows = []
    best_name = None
    best_acc = -1.0

    for name, clf in models.items():
        pipe = Pipeline(steps=[('preprocess', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rep = classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred))

        # Save model per algorithm and potentially best
        model_path = MODELS_DIR / f'fert_model_{name}.joblib'
        joblib.dump(pipe, model_path)
        # Save scaler if exists
        try:
            scaler = pipe.named_steps['preprocess'].named_transformers_.get('num')
            if scaler is not None:
                joblib.dump(scaler, MODELS_DIR / 'fert_scaler.joblib')
        except Exception:
            pass

        # Save report and confusion matrix
        (FIG_DIR / f'report_fert_{name}.txt').write_text(rep, encoding='utf-8')
        plot_confusion(
            y_test, y_pred,
            labels=list(le.classes_),
            title=f'Fertilizer Confusion ({name})',
            out_path=FIG_DIR / f'confusion_fert_{name}.png'
        )

        rows.append({'model': name, 'accuracy': float(acc)})
        if acc > best_acc:
            best_acc = acc
            best_name = name

    # Save metrics table
    metrics_df = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
    metrics_path = FIG_DIR / 'fert_model_eval_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    ts = int(time.time())
    metrics_df.to_csv(FIG_DIR / f'fert_model_eval_metrics_{ts}.csv', index=False)

    # Save best model alias + meta
    if best_name is not None:
        best_src = MODELS_DIR / f'fert_model_{best_name}.joblib'
        best_dst = MODELS_DIR / 'fert_best_model.joblib'
        joblib.dump(joblib.load(best_src), best_dst)
        meta = {
            'best_model': best_name,
            'accuracy': best_acc,
            'data_path': str(data_path.resolve()),
            'timestamp': ts,
            'num_cols': num_cols,
            'cat_cols': cat_cols,
        }
        (MODELS_DIR / 'fert_best_model_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print('Data path:', data_path)
    print('Metrics:\n', metrics_df.to_string(index=False))
    print(f'Best model: {best_name} (accuracy={best_acc:.4f})')
    print('Artifacts saved under models/ and figures/.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
