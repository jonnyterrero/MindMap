"""Hand-checked golden tests for the pure feature transforms."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from mindmap_ml.features.engineering import add_forward_labels, build_features


def _frame(anxiety: list, migraine: list | None = None) -> pd.DataFrame:
    n = len(anxiety)
    base = date(2025, 1, 1)
    return pd.DataFrame(
        {
            "user_id": ["u1"] * n,
            "entry_date": [base + timedelta(days=i) for i in range(n)],
            "anxiety": anxiety,
            "sleep_minutes": [400 + 10 * i for i in range(n)],
            "migraine": migraine if migraine is not None else [False] * n,
        }
    )


def test_lags_rolling_deltas_and_missing() -> None:
    df = build_features(_frame([1.0, 2.0, 3.0, 4.0]))

    assert df["anxiety_lag1"].tolist()[1:] == [1.0, 2.0, 3.0]
    assert np.isnan(df["anxiety_lag1"].iloc[0])

    # rolling mean, window 3, min_periods=1
    assert df["anxiety_rollmean3"].tolist() == [1.0, 1.5, 2.0, 3.0]

    # delta vs t-1
    assert df["anxiety_delta1"].tolist()[1:] == [1.0, 1.0, 1.0]
    assert np.isnan(df["anxiety_delta1"].iloc[0])

    assert df["anxiety_missing"].tolist() == [0, 0, 0, 0]


def test_missingness_flag_detects_nan() -> None:
    df = build_features(_frame([1.0, np.nan, 3.0, 4.0]))
    assert df["anxiety_missing"].tolist() == [0, 1, 0, 0]


def test_forward_labels_positive_and_tail_nan() -> None:
    df = add_forward_labels(_frame([1.0, 8.0, 1.0, 1.0]), horizon=1)
    labels = df["label_anxiety"].tolist()
    # row0 sees anxiety[1]=8 (>=7) -> 1; rows1,2 -> 0; tail row3 -> NaN
    assert labels[0] == 1.0
    assert labels[1] == 0.0
    assert labels[2] == 0.0
    assert np.isnan(labels[3])
