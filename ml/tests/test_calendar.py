from datetime import date

import pandas as pd

from mindmap_ml.features.calendar import logging_stats, to_daily_calendar


def _frame(dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame({"user_id": ["u1"] * len(dates), "entry_date": dates, "anxiety": [3.0] * len(dates)})


def test_reindex_fills_gaps_with_logged_flag() -> None:
    df = _frame([date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 4)])  # gap on the 3rd
    cal = to_daily_calendar(df)
    assert len(cal) == 4  # 01-01 .. 01-04 inclusive
    assert cal["logged"].tolist() == [True, True, False, True]
    # un-logged day has NaN signal
    assert pd.isna(cal.loc[cal["entry_date"] == date(2025, 1, 3), "anxiety"].iloc[0])


def test_logging_stats_adherence_and_streaks() -> None:
    df = _frame([date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 4)])
    st = logging_stats(df)[0]
    assert st.span_days == 4
    assert st.logged_days == 3
    assert st.adherence == 0.75
    assert st.longest_streak == 2  # 01-01, 01-02
    assert st.current_streak == 1  # 01-04 only (01-03 missing breaks it)
