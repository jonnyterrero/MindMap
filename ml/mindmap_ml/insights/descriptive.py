"""Tier-0 descriptive analytics — per user, no learning.

This is the core of the "here's a pattern in your data" product and the most
honest thing to ship at n-of-1: trends with bands, and **conditional base rates**
("on the day after a short-sleep night, your anxiety was high X% of the time vs
Y% overall"). Everything is explainable, clinician-friendly, and abstains below a
minimum sample size. Patterns, never causes.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .correlations import METRICS

DATE_COL = "entry_date"
_LABEL = dict(METRICS)
_LABEL.update({"migraine": "Migraine", "mania": "Mania", "sleep_minutes": "Sleep duration"})

# (metric, op, threshold). bool is a subclass of int, so migraine == True fits.
Condition = tuple[str, str, "float | int"]
_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq,
}


def _label(metric: str) -> str:
    return _LABEL.get(metric, metric.replace("_", " "))


def _mask(series: pd.Series, op: str, thr: float | int) -> pd.Series:
    return _OPS[op](series, thr).fillna(False).astype(bool)


def _cond_text(cond: Condition) -> str:
    metric, op, thr = cond
    if metric == "sleep_minutes" and op in ("<", "<="):
        return f"short sleep ({op} {float(thr) / 60:.1f}h)"
    human = {">=": "high", ">": "high", "<=": "low", "<": "low", "==": "a"}.get(op, op)
    return f"{human} {_label(metric).lower()}"


@dataclass
class MetricTrend:
    metric: str
    label: str
    n: int
    current: float  # EWMA level (most recent)
    mean: float
    direction: str  # rising | falling | stable
    statement: str


def trend(df: pd.DataFrame, metric: str, *, window: int = 30, halflife: int = 7, min_obs: int = 5) -> MetricTrend | None:
    if metric not in df.columns:
        return None
    s = df.sort_values(DATE_COL)[metric].dropna().astype(float)
    if len(s) < min_obs:
        return None  # abstain on thin data
    recent = s.tail(window)
    ewma = recent.ewm(halflife=halflife).mean()
    std = float(recent.std(ddof=1)) if len(recent) > 1 else 0.0
    change = float(ewma.iloc[-1] - ewma.iloc[0])
    deadband = 0.25 * std
    direction = "stable" if abs(change) <= deadband or std == 0 else ("rising" if change > 0 else "falling")
    return MetricTrend(
        metric=metric,
        label=_label(metric),
        n=int(len(recent)),
        current=round(float(ewma.iloc[-1]), 2),
        mean=round(float(recent.mean()), 2),
        direction=direction,
        statement=f"Your {_label(metric).lower()} has been {direction} lately (recent average {recent.mean():.1f}).",
    )


@dataclass
class ConditionalRate:
    trigger: Condition
    outcome: Condition
    lag: int
    n_trigger: int
    hits: int
    rate: float
    baseline: float
    lift: float | None
    statement: str


def conditional_rate(
    df: pd.DataFrame,
    trigger: Condition,
    outcome: Condition,
    *,
    lag: int = 1,
    min_trigger: int = 5,
) -> ConditionalRate | None:
    """P(outcome on day t+lag | trigger on day t), computed on the continuous
    calendar so only logged outcome days count. Returns None below ``min_trigger``.
    """
    from ..features.calendar import to_daily_calendar

    cal = to_daily_calendar(df).sort_values(DATE_COL).reset_index(drop=True)
    if trigger[0] not in cal.columns or outcome[0] not in cal.columns:
        return None

    trig = _mask(cal[trigger[0]], trigger[1], trigger[2]) & cal["logged"]
    out_event = _mask(cal[outcome[0]], outcome[1], outcome[2])
    out_logged = cal["logged"]

    fut_event = out_event.shift(-lag)
    fut_logged = out_logged.shift(-lag).fillna(False).astype(bool)

    eligible = trig & fut_logged
    n_trigger = int(eligible.sum())
    if n_trigger < min_trigger:
        return None

    hits = int((eligible & fut_event.fillna(False)).sum())
    rate = hits / n_trigger

    base_days = int(out_logged.sum())
    base_hits = int((out_event & out_logged).sum())
    baseline = (base_hits / base_days) if base_days else 0.0
    lift = round(rate / baseline, 2) if baseline > 0 else None

    days = "day" if lag == 1 else f"{lag} days"
    stmt = (
        f"On the {days} after {_cond_text(trigger)}, {_cond_text(outcome)} occurred "
        f"{rate * 100:.0f}% of the time ({hits} of {n_trigger}) — vs {baseline * 100:.0f}% overall. "
        f"A possible pattern, not a cause."
    )
    return ConditionalRate(trigger, outcome, lag, n_trigger, hits, round(rate, 3), round(baseline, 3), lift, stmt)
