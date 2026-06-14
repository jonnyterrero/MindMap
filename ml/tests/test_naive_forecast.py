from datetime import date, timedelta

import pandas as pd

from mindmap_ml.insights.naive_forecast import next_event_probability, persistence_forecast


def _days(n: int) -> list[date]:
    return [date(2025, 1, 1) + timedelta(days=i) for i in range(n)]


def test_next_day_probability_tracks_recent_rate() -> None:
    # 10 days, anxiety high on 3 of them -> smoothed daily rate ~ (3+1)/(10+2) = 0.33
    n = 10
    anx = [8.0 if i < 3 else 2.0 for i in range(n)]
    df = pd.DataFrame({"user_id": ["u1"] * n, "entry_date": _days(n), "anxiety": anx})
    f = next_event_probability(df, ("anxiety", ">=", 7), horizon=1)
    assert not f.abstained
    assert f.probability is not None and 0.2 <= f.probability <= 0.45
    assert f.method == "recent_daily_rate"


def test_abstains_on_thin_history() -> None:
    n = 4
    df = pd.DataFrame({"user_id": ["u1"] * n, "entry_date": _days(n), "anxiety": [8.0, 8, 2, 2]})
    f = next_event_probability(df, ("anxiety", ">=", 7), horizon=1)
    assert f.abstained and f.probability is None


def test_weekly_horizon_returns_probability() -> None:
    n = 20
    anx = [8.0 if i % 4 == 0 else 2.0 for i in range(n)]
    df = pd.DataFrame({"user_id": ["u1"] * n, "entry_date": _days(n), "anxiety": anx})
    f = next_event_probability(df, ("anxiety", ">=", 7), horizon=7)
    assert not f.abstained
    assert f.probability is not None and 0.0 < f.probability <= 1.0
    assert "7d" in f.method


def test_persistence_forecast_level_and_band() -> None:
    n = 12
    mood = [1.0, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    df = pd.DataFrame({"user_id": ["u1"] * n, "entry_date": _days(n), "mood_valence": mood})
    lf = persistence_forecast(df, "mood_valence")
    assert not lf.abstained
    assert lf.predicted is not None and 0.0 <= lf.predicted <= 3.0
    assert lf.band >= 0.0
