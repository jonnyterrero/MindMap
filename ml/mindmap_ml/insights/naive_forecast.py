"""Naive, robust forecasters for n-of-1 — the right "prediction" at small data.

No learned model. Next-day / next-week event probability is the user's own recent
**smoothed base rate** (Laplace-smoothed), and a continuous metric's next value is
its EWMA level. These are honest, calibrated-by-construction, explainable, and they
abstain below a minimum number of logged days. They are the cold-start fallback and
the Tier-0 baseline the eval harness must beat before any learned model ships.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .descriptive import Condition, _cond_text, _mask

DATE_COL = "entry_date"


@dataclass
class EventForecast:
    outcome: Condition
    horizon: int  # days
    probability: float | None  # None when abstaining
    method: str
    n_obs: int
    abstained: bool
    statement: str


def next_event_probability(
    df: pd.DataFrame,
    outcome: Condition,
    *,
    horizon: int = 1,
    window: int = 28,
    min_obs: int = 7,
    laplace: float = 1.0,
) -> EventForecast:
    """P(event within the next ``horizon`` days), from the user's recent rate.

    horizon=1 → next-day daily rate. horizon=7 → empirical "any event in a 7-day
    window" when enough history exists, else an independence approximation.
    """
    from ..features.calendar import to_daily_calendar

    cal = to_daily_calendar(df).sort_values(DATE_COL)
    logged = cal["logged"]
    event = _mask(cal[outcome[0]], outcome[1], outcome[2]) & logged
    obs = cal.loc[logged]
    n_obs = int(len(obs))
    if n_obs < min_obs or outcome[0] not in cal.columns:
        return EventForecast(
            outcome, horizon, None, "abstain", n_obs, True,
            "Not enough consistent data yet to estimate this reliably.",
        )

    ev = event.loc[logged].astype(float).tail(window)
    daily_rate = (ev.sum() + laplace) / (len(ev) + 2 * laplace)  # Laplace-smoothed

    if horizon == 1:
        prob = daily_rate
        method = "recent_daily_rate"
    else:
        # empirical weekly-window rate if we have enough windows, else independence approx
        windows = event.loc[logged].rolling(horizon).max().dropna()
        if len(windows) >= min_obs:
            prob = float((windows.sum() + laplace) / (len(windows) + 2 * laplace))
            method = f"empirical_{horizon}d_window"
        else:
            prob = float(1 - (1 - daily_rate) ** horizon)
            method = f"independence_approx_{horizon}d"

    span = "tomorrow" if horizon == 1 else f"the next {horizon} days"
    stmt = (
        f"Based on your recent logs, the chance of {_cond_text(outcome)} {span} is about "
        f"{prob * 100:.0f}%. This is an estimate from your own history, not a diagnosis."
    )
    return EventForecast(outcome, horizon, round(float(prob), 3), method, n_obs, False, stmt)


@dataclass
class LevelForecast:
    metric: str
    predicted: float | None
    band: float  # +/- one std of recent values
    n_obs: int
    abstained: bool
    statement: str


def persistence_forecast(
    df: pd.DataFrame, metric: str, *, window: int = 14, halflife: int = 5, min_obs: int = 5
) -> LevelForecast:
    """Next-value forecast for a continuous metric = its EWMA level, with a band."""
    if metric not in df.columns:
        return LevelForecast(metric, None, 0.0, 0, True, "Not enough data yet.")
    s = df.sort_values(DATE_COL)[metric].dropna().astype(float)
    if len(s) < min_obs:
        return LevelForecast(metric, None, 0.0, int(len(s)), True, "Not enough data yet.")
    recent = s.tail(window)
    pred = float(recent.ewm(halflife=halflife).mean().iloc[-1])
    band = float(recent.std(ddof=1)) if len(recent) > 1 else 0.0
    return LevelForecast(
        metric, round(pred, 2), round(band, 2), int(len(recent)), False,
        f"Your {metric.replace('_', ' ')} is trending around {pred:.1f} (±{band:.1f}).",
    )
