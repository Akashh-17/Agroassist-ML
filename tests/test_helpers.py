import pandas as pd
import pytest
from src.helpers import clean_data, build_features, validate_columns


def sample_df():
    return pd.DataFrame(
        {
            "N": [10, 10],
            "P": [5, 5],
            "K": [3, 3],
            "temperature": [25, 25],
            "humidity": [60, 60],
            "ph": [6.2, 6.2],
            "rainfall": [120, 120],
            "label": ["Rice", "Rice"],
        }
    )


def test_validate_columns_ok():
    df = sample_df()
    validate_columns(df)


def test_clean_data_duplicates():
    df = sample_df()
    out = clean_data(pd.concat([df, df]))
    # clean_data drops duplicates, so length should be 1
    assert len(out) == 1


def test_build_features_columns_added():
    df = sample_df()
    df2 = build_features(df)
    assert "np_ratio" in df2.columns
    assert "nk_ratio" in df2.columns
