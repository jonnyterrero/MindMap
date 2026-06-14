"""Continuous daily calendar with explicit missingness.

The raw frame has one row per *logged* day (gaps are simply absent). For honest
descriptive statistics — adherence, streaks, "X of the last N days" — gaps must
be *visible*, not silently dropped. This reindexes each user to a continuous
daily calendar (min→max logged date), marking each day ``logged`` True/False and
leaving signal columns NaN on un-logged days.

Pure functions, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

USER_COL = "user_id"
DATE_COL = "entry_date"


def to_daily_calendar(df: pd.DataFrame, *, user_col: str = USER_COL, date_col: str = DATE_COL) -> pd.DataFrame:
    """Reindex each user to a gap-free daily calendar with a ``logged`` flag."""
    if df.empty:
        out = df.copy()
        out["logged"] = pd.Series(dtype=bool)
        return out

    parts: list[pd.DataFrame] = []
    for uid, g in df.groupby(user_col, sort=False):
        g = g.copy()
        g["_logged"] = True
        idx = pd.to_datetime(g[date_col])
        full = pd.date_range(idx.min(), idx.max(), freq="D")
        g = g.set_index(idx).reindex(full)
        g[user_col] = uid
        g[date_col] = [ts.date() for ts in full]
        g["logged"] = g["_logged"].fillna(False).astype(bool)
        g = g.drop(columns=["_logged"]).reset_index(drop=True)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


@dataclass
class LoggingStats:
    user_id: str
    span_days: int  # calendar days from first to last log inclusive
    logged_days: int
    adherence: float  # logged_days / span_days
    current_streak: int  # consecutive logged days ending at the last calendar day
    longest_streak: int


def _streaks(logged: list[bool]) -> tuple[int, int]:
    longest = cur = 0
    for v in logged:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    # current streak = run length ending at the final day
    current = 0
    for v in reversed(logged):
        if not v:
            break
        current += 1
    return current, longest


def logging_stats(df: pd.DataFrame, *, user_col: str = USER_COL, date_col: str = DATE_COL) -> list[LoggingStats]:
    """Per-user adherence + streaks, computed on the continuous calendar."""
    cal = to_daily_calendar(df, user_col=user_col, date_col=date_col)
    stats: list[LoggingStats] = []
    for uid, g in cal.groupby(user_col, sort=False):
        g = g.sort_values(date_col)
        logged = g["logged"].tolist()
        span = len(logged)
        n_logged = int(sum(logged))
        current, longest = _streaks(logged)
        stats.append(
            LoggingStats(
                user_id=str(uid),
                span_days=span,
                logged_days=n_logged,
                adherence=round(n_logged / span, 3) if span else 0.0,
                current_streak=current,
                longest_streak=longest,
            )
        )
    return stats
