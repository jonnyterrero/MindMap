"""Multivariate drivers — "what might be driving this?".

A regularized (Ridge) regression of a target metric on the *lagged* values of
other metrics, with standardized inputs so coefficients are comparable. This goes
beyond pairwise correlation to estimate each candidate's contribution holding the
others fixed. Conservative: stays silent (returns ``[]``) below a minimum sample
size, and only reports coefficients above a small magnitude. Possible patterns,
never causes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATE_COL = "entry_date"

CANDIDATE_METRICS: list[str] = [
    "sleep_minutes",
    "sleep_quality",
    "anxiety",
    "depression",
    "mood_valence",
    "focus",
    "productivity",
    "body_pain",
    "pressure_change",
    "humidity",
]

_MIN_SAMPLES = 20
_MIN_ABS_COEF = 0.05


@dataclass
class Driver:
    feature: str
    lag: int
    weight: float  # standardized Ridge coefficient
    direction: str  # positive | negative
    statement: str


def drivers_for(
    df: pd.DataFrame,
    target: str,
    *,
    lag: int = 1,
    candidates: list[str] | None = None,
    top_k: int = 5,
    min_samples: int = _MIN_SAMPLES,
    alpha: float = 1.0,
) -> list[Driver]:
    """Top standardized drivers of ``target`` from other metrics at ``t-lag``.

    Returns ``[]`` when there isn't enough data — abstention applies to insights
    too.
    """
    if target not in df.columns:
        return []
    df = df.sort_values(DATE_COL) if DATE_COL in df.columns else df
    feats = [c for c in (candidates or CANDIDATE_METRICS) if c in df.columns and c != target]
    if not feats:
        return []

    x = df[feats].shift(lag)
    y = df[target]
    frame = pd.concat([x, y.rename("__y__")], axis=1).dropna(subset=["__y__"])
    # need the target and at least some feature signal
    frame = frame.dropna(how="all", subset=feats)
    if len(frame) < min_samples:
        return []

    x_mat = frame[feats].to_numpy(dtype=float)
    y_vec = frame["__y__"].to_numpy(dtype=float)
    if np.nanstd(y_vec) == 0:
        return []

    model = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(x_mat, y_vec)
    coefs = model.named_steps["ridge"].coef_

    drivers: list[Driver] = []
    for name, coef in zip(feats, coefs, strict=True):
        if abs(coef) < _MIN_ABS_COEF:
            continue
        direction = "positive" if coef > 0 else "negative"
        days = "day" if lag == 1 else "days"
        rel = "higher" if coef > 0 else "lower"
        drivers.append(
            Driver(
                feature=name,
                lag=lag,
                weight=round(float(coef), 3),
                direction=direction,
                statement=(
                    f"{name} about {lag} {days} earlier is a possible driver of {rel} {target} "
                    f"(standardized weight {coef:+.2f}). This is a pattern, not a cause."
                ),
            )
        )
    drivers.sort(key=lambda d: abs(d.weight), reverse=True)
    return drivers[:top_k]
