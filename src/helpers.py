from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

REQUIRED_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower()
    for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
    df = df.drop_duplicates()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if all(c in df.columns for c in ["N", "P", "K"]):
        df["np_ratio"] = (df["N"] + 1) / (df["P"] + 1)
        df["nk_ratio"] = (df["N"] + 1) / (df["K"] + 1)
    if "ph" in df.columns:
        df["ph_bucket"] = pd.cut(
            df["ph"],
            bins=[0, 5.5, 6.5, 7.5, 14],
            labels=["acidic", "slightly_acidic", "neutral", "alkaline"],
        )
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
