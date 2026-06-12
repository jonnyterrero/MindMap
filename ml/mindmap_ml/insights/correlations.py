"""Personalized correlations — contemporaneous and **lagged**.

Extends the app's conservative Pearson approach (frontend/lib/correlation-engine.ts)
with lead/lag analysis: does metric A tend to *precede* metric B by k days? Every
result is phrased as a *possible pattern*, never a cause, and the same minimum
sample size / minimum |r| conservatism is preserved.

Operates on a single user's chronological daily frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    CORR_MAX_RESULTS,
    CORR_MIN_ABS_R,
    CORR_MIN_SAMPLE_SIZE,
    CORR_STRENGTH_MODERATE,
    CORR_STRENGTH_STRONG,
)

DATE_COL = "entry_date"

# Same neutral metric set as the TS engine (direction described neutrally).
METRICS: list[tuple[str, str]] = [
    ("sleep_minutes", "Sleep duration"),
    ("sleep_quality", "Sleep quality"),
    ("mood_valence", "Mood"),
    ("anxiety", "Anxiety"),
    ("depression", "Depression"),
    ("focus", "Focus"),
    ("productivity", "Productivity"),
    ("migraine_intensity", "Migraine intensity"),
    ("body_pain", "Body pain"),
    ("pressure", "Barometric pressure"),
    ("humidity", "Humidity"),
    ("temp_max", "Temperature"),
]
_LABEL = dict(METRICS)


@dataclass
class Correlation:
    a_key: str
    a_label: str
    b_key: str
    b_label: str
    r: float
    n: int
    strength: str
    direction: str  # positive | negative
    lag: int  # 0 = contemporaneous; k>=1 = a leads b by k days
    statement: str


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    den = float(np.sqrt((dx * dx).sum() * (dy * dy).sum()))
    if den == 0:
        return None
    return float((dx * dy).sum() / den)


def _strength(abs_r: float) -> str:
    if abs_r >= CORR_STRENGTH_STRONG:
        return "strong"
    if abs_r >= CORR_STRENGTH_MODERATE:
        return "moderate"
    return "weak"


def _aligned(df: pd.DataFrame, a: str, b: str, lag: int) -> tuple[np.ndarray, np.ndarray]:
    sa = df[a].shift(lag) if lag else df[a]  # a at t-lag
    sb = df[b]
    pair = pd.concat([sa, sb], axis=1).dropna()
    return pair.iloc[:, 0].to_numpy(dtype=float), pair.iloc[:, 1].to_numpy(dtype=float)


def _statement(a: str, b: str, r: float, n: int, strength: str, lag: int) -> str:
    if lag == 0:
        rel = (
            f"{a} and {b} tended to rise and fall together"
            if r > 0
            else f"when {a} was higher, {b} tended to be lower"
        )
        return f"{rel} — a {strength} possible pattern across {n} days. This is an association, not a cause."
    days = "day" if lag == 1 else "days"
    dir_word = "higher" if r > 0 else "lower"
    return (
        f"{a} changes tended to precede {dir_word} {b} about {lag} {days} later "
        f"— a {strength} possible pattern across {n} paired days. This is an association, not a cause."
    )


def compute_correlations(
    df: pd.DataFrame,
    *,
    min_sample: int = CORR_MIN_SAMPLE_SIZE,
    min_abs_r: float = CORR_MIN_ABS_R,
    max_results: int = CORR_MAX_RESULTS,
) -> list[Correlation]:
    """Contemporaneous correlations (lag 0), strongest first."""
    cols = [k for k, _ in METRICS if k in df.columns]
    results: list[Correlation] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            xs, ys = _aligned(df, a, b, lag=0)
            if len(xs) < min_sample:
                continue
            r = _pearson(xs, ys)
            if r is None or abs(r) < min_abs_r:
                continue
            strength = _strength(abs(r))
            results.append(
                Correlation(
                    a, _LABEL[a], b, _LABEL[b], round(r, 2), len(xs), strength,
                    "positive" if r > 0 else "negative", 0,
                    _statement(_LABEL[a], _LABEL[b], r, len(xs), strength, 0),
                )
            )
    results.sort(key=lambda c: abs(c.r), reverse=True)
    return results[:max_results]


def compute_lagged_correlations(
    df: pd.DataFrame,
    *,
    max_lag: int = 3,
    min_sample: int = CORR_MIN_SAMPLE_SIZE,
    min_abs_r: float = CORR_MIN_ABS_R,
    max_results: int = CORR_MAX_RESULTS,
) -> list[Correlation]:
    """Lead/lag correlations: A at t-k vs B at t (A leads B). Keeps the strongest
    lag per ordered pair. Conservative thresholds; possible patterns only."""
    df = df.sort_values(DATE_COL) if DATE_COL in df.columns else df
    cols = [k for k, _ in METRICS if k in df.columns]
    best: dict[tuple[str, str], Correlation] = {}
    for a in cols:
        for b in cols:
            if a == b:
                continue
            for lag in range(1, max_lag + 1):
                xs, ys = _aligned(df, a, b, lag=lag)
                if len(xs) < min_sample:
                    continue
                r = _pearson(xs, ys)
                if r is None or abs(r) < min_abs_r:
                    continue
                key = (a, b)
                if key in best and abs(best[key].r) >= abs(r):
                    continue
                strength = _strength(abs(r))
                best[key] = Correlation(
                    a, _LABEL[a], b, _LABEL[b], round(r, 2), len(xs), strength,
                    "positive" if r > 0 else "negative", lag,
                    _statement(_LABEL[a], _LABEL[b], r, len(xs), strength, lag),
                )
    out = sorted(best.values(), key=lambda c: abs(c.r), reverse=True)
    return out[:max_results]
