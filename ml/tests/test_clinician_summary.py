"""Phase-6 integrative deliverable: the clinician summary must be evidence-cited,
abstain on thin data, surface safety flags, and never emit ungated text.
"""

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from mindmap_ml.labels.instruments import score_phq9
from mindmap_ml.reports.clinician_summary import build_clinician_summary
from mindmap_ml.safety.gate import check_output
from mindmap_ml.synthetic.generator import generate_dataset


@pytest.fixture(scope="module")
def rich_user() -> pd.DataFrame:
    df = generate_dataset(seed=0, n_days=150)
    return df[df["user_id"] == "anxiety_after_poor_sleep_00"].copy()


def _all_statements(summary) -> list[str]:
    out = [t["statement"] for t in summary.trajectories]
    out += [p.statement for p in summary.detected_patterns]
    out += [w["statement"] for w in summary.watch_items]
    return out


def _risk_claim_statements(summary) -> list[str]:
    # patterns + watch items are risk claims (must carry uncertainty framing);
    # trajectories are descriptive of the user's own data (framing not required).
    return [p.statement for p in summary.detected_patterns] + [w["statement"] for w in summary.watch_items]


def test_rich_user_summary_is_grounded(rich_user) -> None:
    s = build_clinician_summary(rich_user)
    assert not s.abstained
    assert s.readiness["ready"] is True
    assert s.trajectories, "should report metric trajectories"
    assert s.detected_patterns, "should detect at least one pattern"
    assert any(p.citations for p in s.detected_patterns), "a grounded pattern must carry citations"
    assert s.watch_items, "should produce next-day/next-week watch items"


def test_no_surfaced_statement_contains_banned_phrasing(rich_user) -> None:
    s = build_clinician_summary(rich_user)
    # banned-phrase check runs regardless of is_risk_claim -> hard safety guarantee
    for stmt in _all_statements(s):
        assert not check_output(stmt, is_risk_claim=False).violations, f"banned phrase in: {stmt!r}"


def test_risk_claims_carry_uncertainty_framing(rich_user) -> None:
    s = build_clinician_summary(rich_user)
    for stmt in _risk_claim_statements(s):
        assert check_output(stmt, is_risk_claim=True).allowed, f"risk claim missing framing: {stmt!r}"


def test_thin_user_abstains_with_countdown() -> None:
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(5)]
    df = pd.DataFrame({
        "user_id": ["u1"] * 5, "entry_date": days,
        "anxiety": [6.0, 7, 5, 8, 6], "sleep_minutes": [400.0] * 5,
        "depression": [3.0] * 5, "mood_valence": [1.0] * 5,
    })
    s = build_clinician_summary(df)
    assert s.abstained is True
    assert s.readiness["ready"] is False
    assert s.readiness["days_remaining"] > 0
    assert s.detected_patterns == []


def test_partial_readiness_shows_trajectories_but_gates_patterns() -> None:
    # >= 7 logged days but < 30 -> not abstained, trajectories shown, but patterns
    # and forecasts are withheld until there's enough data to be reliable.
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(12)]
    df = pd.DataFrame({
        "user_id": ["u1"] * 12, "entry_date": days,
        "anxiety": [3.0, 4, 5, 6, 5, 4, 3, 4, 5, 6, 5, 4], "sleep_minutes": [420.0] * 12,
        "depression": [2.0] * 12, "mood_valence": [1.0] * 12, "focus": [6.0] * 12,
    })
    s = build_clinician_summary(df)
    assert s.abstained is False
    assert s.readiness["ready"] is False
    assert s.trajectories  # descriptive trajectories still shown
    assert s.detected_patterns == []  # patterns gated until ~30 days
    assert s.watch_items == []


def test_phq9_item9_raises_crisis(rich_user) -> None:
    phq9 = score_phq9([0, 0, 0, 0, 0, 0, 0, 0, 2])  # item 9 positive
    s = build_clinician_summary(rich_user, phq9=phq9)
    assert "phq9_item9_positive" in s.safety_flags
    assert s.crisis is not None and s.crisis["severity"] == "critical"
    assert s.crisis["resources"]


def test_summary_is_json_serializable(rich_user) -> None:
    json.dumps(build_clinician_summary(rich_user).to_dict())
