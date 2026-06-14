from datetime import date, timedelta

import pandas as pd

from mindmap_ml.eval.walk_forward import walk_forward_event


def _frame(n: int, event_every: int) -> pd.DataFrame:
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(n)]
    anx = [8.0 if i % event_every == 0 else 2.0 for i in range(n)]
    return pd.DataFrame({"user_id": ["u1"] * n, "entry_date": days, "anxiety": anx})


def test_walk_forward_is_calibrated_on_stable_rate() -> None:
    # stable ~33% base rate, no temporal structure the rate-forecaster can't capture
    res = walk_forward_event(_frame(60, 3), ("anxiety", ">=", 7), horizon=1, min_history=7)
    assert res.n_points > 0 and res.n_covered > 0
    assert 0.0 <= res.coverage <= 1.0
    assert res.coverage < 1.0  # early thin-history points abstain
    assert abs(res.base_rate - 1 / 3) < 0.12
    assert res.ece < 0.2  # recent-rate forecaster is well-calibrated on a stable rate
    assert res.brier == res.brier  # not NaN


def test_walk_forward_abstains_when_no_history() -> None:
    res = walk_forward_event(_frame(6, 2), ("anxiety", ">=", 7), horizon=1, min_history=7)
    # only 6 days, need 7 history -> everything abstains -> no covered points
    assert res.n_covered == 0
