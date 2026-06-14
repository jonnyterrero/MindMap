from datetime import date, timedelta

import pandas as pd

from mindmap_ml.insights.descriptive import conditional_rate, trend


def _days(n: int) -> list[date]:
    return [date(2025, 1, 1) + timedelta(days=i) for i in range(n)]


def test_trend_detects_rising() -> None:
    df = pd.DataFrame({"user_id": ["u1"] * 8, "entry_date": _days(8), "anxiety": [1.0, 2, 3, 4, 5, 6, 7, 8]})
    t = trend(df, "anxiety")
    assert t is not None and t.direction == "rising"


def test_trend_stable_when_flat() -> None:
    df = pd.DataFrame({"user_id": ["u1"] * 8, "entry_date": _days(8), "anxiety": [5.0] * 8})
    t = trend(df, "anxiety")
    assert t is not None and t.direction == "stable"


def test_trend_abstains_on_thin_data() -> None:
    df = pd.DataFrame({"user_id": ["u1"] * 3, "entry_date": _days(3), "anxiety": [5.0, 6, 7]})
    assert trend(df, "anxiety") is None


def test_conditional_rate_recovers_sleep_to_anxiety() -> None:
    # 12 continuous days; short sleep on even days, high anxiety the day after each.
    n = 12
    sleep = [300.0 if i % 2 == 0 else 480.0 for i in range(n)]
    anx = [8.0 if i % 2 == 1 else 2.0 for i in range(n)]
    df = pd.DataFrame({"user_id": ["u1"] * n, "entry_date": _days(n), "sleep_minutes": sleep, "anxiety": anx})
    cr = conditional_rate(df, ("sleep_minutes", "<", 360), ("anxiety", ">=", 7), lag=1)
    assert cr is not None
    assert cr.n_trigger >= 5
    assert cr.rate == 1.0  # every short-sleep night was followed by high anxiety
    assert cr.lift is not None and cr.lift > 1
    assert "not a cause" in cr.statement


def test_conditional_rate_abstains_without_enough_triggers() -> None:
    n = 10
    sleep = [480.0] * n
    sleep[0] = 300.0  # only one trigger day
    df = pd.DataFrame({"user_id": ["u1"] * n, "entry_date": _days(n), "sleep_minutes": sleep, "anxiety": [3.0] * n})
    assert conditional_rate(df, ("sleep_minutes", "<", 360), ("anxiety", ">=", 7)) is None
