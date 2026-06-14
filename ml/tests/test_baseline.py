"""Golden tests: the Python baseline must reproduce the TS engines' numbers.

Expected values are derived by hand from insights-engine.ts / prediction-engine.ts.
"""

from datetime import date

from mindmap_ml.config import BASELINE_MODEL_VERSION
from mindmap_ml.eval.baseline import (
    RuleBaselineModel,
    compute_migraine_risk,
    compute_mood_trend,
    compute_predictions,
)
from mindmap_ml.models.base import PredictionWindow


def test_migraine_risk_reproduces_ts() -> None:
    entries = [{"sleep_minutes": 300, "anxiety": 8, "depression": 7, "migraine": False}]
    r = compute_migraine_risk(entries)
    # 30 (sleep<6) + 25 (anx>=7) + 15 (dep>=7) = 70
    assert r["score"] == 70
    assert r["risk_level"] == "high"
    assert r["reasons"][0] == "Sleep was only 5.0h (< 6h)"
    assert "Anxiety is high (8/10)" in r["reasons"]


def test_migraine_risk_empty_is_unknown() -> None:
    r = compute_migraine_risk([])
    assert r["risk_level"] == "unknown"
    assert r["score"] == 0


def test_mood_trend_reproduces_ts() -> None:
    e = {"anxiety": 7, "depression": 7, "focus": 2, "productivity": 2}
    r = compute_mood_trend([e, e, e])
    # 30 + 30 + 15 + 15 = 90
    assert r["score"] == 90
    assert r["risk_level"] == "concerning"


def test_mood_trend_needs_three_days() -> None:
    r = compute_mood_trend([{"anxiety": 9}, {"anxiety": 9}])
    assert r["risk_level"] == "unknown"


def test_predictions_anxiety_score_and_confidence() -> None:
    entries = [{"anxiety": 6, "sleep_minutes": 480, "migraine": False}] * 14
    preds = {p["prediction_type"]: p for p in compute_predictions({"entries": entries})}
    anx = preds["anxiety"]
    assert anx["risk_score"] == 0.6
    assert anx["risk_level"] == "moderate"
    assert anx["confidence"] == 0.7
    assert anx["model_version"] == BASELINE_MODEL_VERSION


def test_predictions_empty_returns_empty() -> None:
    assert compute_predictions({"entries": []}) == []


def test_pressure_drop_raises_migraine_via_weather() -> None:
    entries = [{"sleep_minutes": 480, "migraine": False}] * 8
    no_weather = {p["prediction_type"]: p for p in compute_predictions({"entries": entries})}
    with_drop = {
        p["prediction_type"]: p
        for p in compute_predictions({"entries": entries, "weather": {"pressure_change": -12}})
    }
    assert with_drop["migraine"]["risk_score"] > no_weather["migraine"]["risk_score"]


def test_model_abstains_under_min_history() -> None:
    model = RuleBaselineModel()
    short = PredictionWindow("u1", date(2025, 1, 5), entries=[{"anxiety": 6}] * 3)
    preds = model.predict(short)
    assert all(p.abstained for p in preds)
    assert all(p.abstain_reason == "insufficient_history" for p in preds)


def test_model_emits_when_enough_history() -> None:
    model = RuleBaselineModel()
    long = PredictionWindow(
        "u1", date(2025, 1, 20), entries=[{"anxiety": 6, "sleep_minutes": 480, "migraine": False}] * 14
    )
    preds = {p.prediction_type: p for p in model.predict(long)}
    assert preds["anxiety"].abstained is False
    assert preds["anxiety"].risk == 0.6


def test_fallback_mode_never_abstains() -> None:
    model = RuleBaselineModel(apply_abstention=False)
    short = PredictionWindow("u1", date(2025, 1, 5), entries=[{"anxiety": 6}] * 2)
    preds = model.predict(short)
    assert all(not p.abstained for p in preds)
