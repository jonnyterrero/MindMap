"""Walk-forward (expanding-window) per-user backtest for the naive forecasters.

The honest evaluation at n-of-1: at each logged day t, predict P(event at t+horizon)
using ONLY data up to and including t, then compare to the realized outcome. We
report calibration (ECE/Brier) and abstention coverage, per outcome. This both
validates the Tier-0 forecaster and sets the bar any future learned model must
beat (calibration first).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.calendar import to_daily_calendar
from ..insights.descriptive import Condition, _mask
from . import metrics

USER_COL = "user_id"
DATE_COL = "entry_date"


@dataclass
class WalkForwardResult:
    outcome: str
    horizon: int
    n_points: int
    n_covered: int
    coverage: float
    base_rate: float
    brier: float
    ece: float
    auroc: float


def _smoothed_rate(events: np.ndarray, window: int, laplace: float) -> float:
    w = events[-window:]
    return float((w.sum() + laplace) / (len(w) + 2 * laplace))


def walk_forward_event(
    df: pd.DataFrame,
    outcome: Condition,
    *,
    horizon: int = 1,
    window: int = 28,
    min_history: int = 7,
    laplace: float = 1.0,
    calibration_bins: int = 10,
) -> WalkForwardResult:
    """Backtest the recent-rate forecaster for one event across all users."""
    preds: list[float] = []
    labels: list[float] = []
    abstained: list[bool] = []

    for _uid, g in df.groupby(USER_COL, sort=False):
        cal = to_daily_calendar(g).sort_values(DATE_COL).reset_index(drop=True)
        logged = cal["logged"].to_numpy(dtype=bool)
        event = (_mask(cal[outcome[0]], outcome[1], outcome[2]).to_numpy(dtype=bool) & logged)
        n = len(cal)

        for t in range(n - horizon):
            if not logged[t] or not logged[t + horizon]:
                continue  # predict on logged days; score only when the target day is logged
            hist_events = event[: t + 1][logged[: t + 1]].astype(float)  # logged-day events up to t
            y_t = float(event[t + horizon])
            if len(hist_events) < min_history:
                abstained.append(True)
                preds.append(float("nan"))
                labels.append(y_t)
                continue
            p_t = _smoothed_rate(hist_events, window, laplace)
            abstained.append(False)
            preds.append(p_t)
            labels.append(y_t)

    ab = np.asarray(abstained, dtype=bool)
    p = np.asarray(preds, dtype=float)
    y = np.asarray(labels, dtype=float)
    covered = ~ab
    p_cov, y_cov = p[covered], y[covered]
    n_covered = int(covered.sum())
    return WalkForwardResult(
        outcome=outcome[0],
        horizon=horizon,
        n_points=int(len(y)),
        n_covered=n_covered,
        coverage=float(covered.mean()) if len(ab) else float("nan"),
        base_rate=float(y_cov.mean()) if n_covered else float("nan"),
        brier=metrics.brier_score(y_cov, p_cov) if n_covered else float("nan"),
        ece=metrics.expected_calibration_error(y_cov, p_cov, calibration_bins) if n_covered else float("nan"),
        auroc=metrics.auroc(y_cov, p_cov) if n_covered else float("nan"),
    )
