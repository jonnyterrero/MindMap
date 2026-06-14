"""Phase 4: lagged correlations + multivariate drivers must recover a known
effect and stay silent on noise — with the TS engine's conservative thresholds.
"""

import pandas as pd
import pytest

from mindmap_ml.insights.correlations import (
    compute_correlations,
    compute_lagged_correlations,
)
from mindmap_ml.insights.drivers import drivers_for
from mindmap_ml.synthetic.generator import generate_dataset


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return generate_dataset(seed=0, n_days=150)


def _user(df: pd.DataFrame, persona: str) -> pd.DataFrame:
    return df[df["user_id"] == f"{persona}_00"].copy()


def test_recovers_known_lagged_sleep_to_anxiety(dataset) -> None:
    user = _user(dataset, "anxiety_after_poor_sleep")
    corrs = compute_lagged_correlations(user, max_lag=3)
    hit = [c for c in corrs if c.a_key == "sleep_minutes" and c.b_key == "anxiety"]
    assert hit, "should recover sleep -> anxiety lead/lag effect"
    assert hit[0].direction == "negative"  # less sleep -> more anxiety next day
    assert "not a cause" in hit[0].statement


def test_silent_on_noise_for_same_pair(dataset) -> None:
    user = _user(dataset, "stable_mood")
    corrs = compute_lagged_correlations(user, max_lag=3)
    hit = [c for c in corrs if c.a_key == "sleep_minutes" and c.b_key == "anxiety"]
    assert not hit, "no real sleep->anxiety effect for the stable persona"


def test_contemporaneous_correlations_are_conservative(dataset) -> None:
    user = _user(dataset, "anxiety_after_poor_sleep")
    corrs = compute_correlations(user)
    assert len(corrs) <= 6
    assert all(abs(c.r) >= 0.3 for c in corrs)
    assert all(c.lag == 0 for c in corrs)


def test_drivers_recover_sleep_for_anxiety(dataset) -> None:
    user = _user(dataset, "anxiety_after_poor_sleep")
    drivers = drivers_for(user, target="anxiety", lag=1)
    by_feat = {d.feature: d for d in drivers}
    assert "sleep_minutes" in by_feat
    assert by_feat["sleep_minutes"].weight < 0  # less sleep -> more anxiety


def test_drivers_stay_silent_on_tiny_data() -> None:
    tiny = pd.DataFrame(
        {
            "entry_date": pd.date_range("2025-01-01", periods=5).date,
            "anxiety": [1, 2, 3, 4, 5],
            "sleep_minutes": [400, 410, 420, 430, 440],
        }
    )
    assert drivers_for(tiny, target="anxiety", lag=1) == []
