"""Deterministic rule baseline — a faithful Python port of the app's TS engines:

  * frontend/lib/insights-engine.ts   (computeMigraineRisk, computeMoodTrend)
  * frontend/lib/prediction-engine.ts (computePredictions, MODEL_VERSION
    "v1_rule_extended")

Thresholds, defaults, rounding and the newest-first window semantics are
preserved verbatim so a golden test can prove byte-for-byte agreement with the
app. This is both the model-to-beat (Phase 3) and the cold-start fallback
(Phase 5). It also wraps the rules in the uniform :class:`Model` interface and
applies the abstention contract.
"""

from __future__ import annotations

import math
from typing import Any

from ..config import BASELINE_MODEL_VERSION, MIN_HISTORY_DAYS
from ..models.base import ContributingFactor, Prediction, PredictionWindow
from ..safety.contract import AbstentionDecision, decide_abstention

Entry = dict[str, Any]


# --------------------------------------------------------------------------- #
# Shared numeric helpers (mirror the TS clamp01 / num / avg).
# --------------------------------------------------------------------------- #
def _clamp01(n: float) -> float:
    return max(0.0, min(1.0, n))


def _num(v: Any) -> float | None:
    """JS ``typeof v === "number"`` equivalent: booleans and NaN are not numbers."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return None if math.isnan(f) else f
    return None


def _mean(nums: list[float]) -> float:
    return sum(nums) / len(nums) if nums else 0.0


def _mean_present(values: list[Any]) -> float:
    present = [n for n in (_num(v) for v in values) if n is not None]
    return _mean(present)


def _num_or(v: Any, default: float) -> float:
    n = _num(v)
    return n if n is not None else float(default)


def _recent(entries: list[Entry]) -> list[Entry]:
    return entries[:7]


# --------------------------------------------------------------------------- #
# Port of insights-engine.ts
# --------------------------------------------------------------------------- #
def _sleep_variance(entries: list[Entry]) -> float:
    sleeps = [n for n in (_num(e.get("sleep_minutes")) for e in entries) if n is not None]
    if len(sleeps) < 2:
        return 0.0
    mean = _mean(sleeps)
    variance = sum((v - mean) ** 2 for v in sleeps) / len(sleeps)  # population (/n), matches TS
    return math.sqrt(variance)


def compute_migraine_risk(entries: list[Entry]) -> dict[str, Any]:
    if not entries:
        return {
            "insight_type": "migraine_risk",
            "risk_level": "unknown",
            "score": 0,
            "reasons": ["Not enough data"],
            "signals": {},
            "recommendation": "Log a few days of data to see predictions.",
        }

    reasons: list[str] = []
    signals: dict[str, Any] = {}
    score = 0

    latest = entries[0]
    sm = _num(latest.get("sleep_minutes"))
    sleep_minutes = sm if sm is not None else 480
    sleep_hrs = sleep_minutes / 60
    anxiety = _num(latest.get("anxiety")) or 0
    depression = _num(latest.get("depression")) or 0

    if sleep_hrs < 6:
        score += 30
        reasons.append(f"Sleep was only {sleep_hrs:.1f}h (< 6h)")
        signals["sleep_hours"] = sleep_hrs
    elif sleep_hrs < 7:
        score += 15
        reasons.append(f"Sleep was {sleep_hrs:.1f}h (< 7h)")
        signals["sleep_hours"] = sleep_hrs

    if anxiety >= 7:
        score += 25
        reasons.append(f"Anxiety is high ({anxiety:g}/10)")
        signals["anxiety"] = anxiety
    elif anxiety >= 5:
        score += 10
        reasons.append(f"Anxiety is moderate ({anxiety:g}/10)")
        signals["anxiety"] = anxiety

    if depression >= 7:
        score += 15
        reasons.append(f"Depression is elevated ({depression:g}/10)")
        signals["depression"] = depression

    recent_migraines = sum(1 for e in entries[:7] if e.get("migraine") is True)
    if recent_migraines >= 3:
        score += 30
        reasons.append(f"{recent_migraines} migraines in the last 7 days")
        signals["recent_migraines"] = recent_migraines
    elif recent_migraines >= 1:
        score += 10
        reasons.append(f"{recent_migraines} migraine(s) in the last 7 days")
        signals["recent_migraines"] = recent_migraines

    sleep_variance = _sleep_variance(entries[:7])
    if sleep_variance > 90:
        score += 15
        reasons.append("Sleep schedule is irregular (high variance)")
        signals["sleep_variance_minutes"] = round(sleep_variance)

    score = min(score, 100)

    if score >= 60:
        risk_level = "high"
        recommendation = "Consider rest, hydration, and avoiding triggers today."
    elif score >= 30:
        risk_level = "moderate"
        recommendation = "Monitor for early migraine signs. Prioritize good sleep tonight."
    else:
        risk_level = "low"
        recommendation = "Looking good! Keep up your current routine."

    if not reasons:
        reasons.append("No risk factors detected")

    return {
        "insight_type": "migraine_risk",
        "risk_level": risk_level,
        "score": score,
        "reasons": reasons,
        "signals": signals,
        "recommendation": recommendation,
    }


def compute_mood_trend(entries: list[Entry]) -> dict[str, Any]:
    if len(entries) < 3:
        return {
            "insight_type": "mood_trend",
            "risk_level": "unknown",
            "score": 0,
            "reasons": ["Need at least 3 days of data"],
            "signals": {},
            "recommendation": "Keep logging daily.",
        }

    recent = _recent(entries)
    # NB: focus/productivity default to 5 when missing — a quirk of the TS engine
    # we reproduce intentionally (productivity is 0..100 but the default is 5).
    avg_anxiety = _mean([_num_or(e.get("anxiety"), 0) for e in recent])
    avg_depression = _mean([_num_or(e.get("depression"), 0) for e in recent])
    avg_focus = _mean([_num_or(e.get("focus"), 5) for e in recent])
    avg_productivity = _mean([_num_or(e.get("productivity"), 5) for e in recent])

    reasons: list[str] = []
    signals: dict[str, Any] = {
        "avgAnxiety": avg_anxiety,
        "avgDepression": avg_depression,
        "avgFocus": avg_focus,
        "avgProductivity": avg_productivity,
    }
    score = 0

    if avg_anxiety >= 6:
        score += 30
        reasons.append(f"Average anxiety is {avg_anxiety:.1f}/10")
    if avg_depression >= 6:
        score += 30
        reasons.append(f"Average depression is {avg_depression:.1f}/10")
    if avg_focus <= 3:
        score += 15
        reasons.append(f"Average focus is low ({avg_focus:.1f}/10)")
    if avg_productivity <= 3:
        score += 15
        reasons.append(f"Average productivity is low ({avg_productivity:.1f}/10)")

    score = min(score, 100)

    if score >= 50:
        risk_level = "concerning"
        recommendation = "Consider discussing recent mood patterns with a therapist."
    elif score >= 20:
        risk_level = "moderate"
        recommendation = "Some mood fluctuation detected. Self-care practices may help."
    else:
        risk_level = "stable"
        recommendation = "Your mood has been steady. Great job maintaining your routines!"

    if not reasons:
        reasons.append("Mood metrics are within healthy ranges")

    return {
        "insight_type": "mood_trend",
        "risk_level": risk_level,
        "score": score,
        "reasons": reasons,
        "signals": signals,
        "recommendation": recommendation,
    }


# --------------------------------------------------------------------------- #
# Port of prediction-engine.ts (v1_rule_extended)
# --------------------------------------------------------------------------- #
PREDICTION_TYPES = ("migraine", "anxiety", "mood", "pain_flare")


def _level_for(score: float) -> str:
    if score > 0.8:
        return "critical"
    if score > 0.6:
        return "high"
    if score > 0.3:
        return "moderate"
    return "low"


def _base_for(ptype: str, entries: list[Entry], factors: list[ContributingFactor]) -> float:
    recent = _recent(entries)
    if ptype == "migraine":
        r = compute_migraine_risk(entries)
        for reason in r["reasons"]:
            factors.append(ContributingFactor("rule_base", 0, reason))
        return r["score"] / 100
    if ptype == "mood":
        r = compute_mood_trend(entries)
        for reason in r["reasons"]:
            factors.append(ContributingFactor("rule_base", 0, reason))
        return r["score"] / 100
    if ptype == "anxiety":
        a = _mean_present([e.get("anxiety") for e in recent])
        if a >= 5:
            factors.append(ContributingFactor("recent_anxiety", a / 10, f"Avg anxiety {a:.1f}/10"))
        return _clamp01(a / 10)
    if ptype == "pain_flare":
        mi = _mean_present([e.get("migraine_intensity") for e in recent])
        ax = _mean_present([e.get("anxiety") for e in recent])
        base = _clamp01((mi / 10) * 0.6 + (ax / 10) * 0.25)
        if mi >= 3:
            factors.append(
                ContributingFactor("recent_pain", (mi / 10) * 0.6, f"Avg migraine intensity {mi:.1f}/10")
            )
        return base
    raise ValueError(f"unknown prediction_type {ptype!r}")


def _apply_wearable_weather(
    ptype: str,
    base: float,
    factors: list[ContributingFactor],
    wearable: dict[str, Any] | None,
    weather: dict[str, Any] | None,
) -> float:
    score = base
    hrv = _num((wearable or {}).get("hrv"))
    sleep_score = _num((wearable or {}).get("sleep_score"))
    resting_hr = _num((wearable or {}).get("resting_hr"))

    if hrv is not None and hrv < 40 and ptype in ("anxiety", "migraine"):
        score += 0.08
        factors.append(ContributingFactor("low_hrv", 0.08, f"HRV {hrv:g}ms (low)"))
    if sleep_score is not None and sleep_score < 60 and ptype != "pain_flare":
        score += 0.15
        factors.append(ContributingFactor("low_sleep_score", 0.15, f"Sleep score {sleep_score:g} (<60)"))
    if resting_hr is not None and resting_hr > 85 and ptype == "pain_flare":
        score += 0.1
        factors.append(ContributingFactor("elevated_resting_hr", 0.1, f"Resting HR {resting_hr:g}bpm (>85)"))
    if hrv is not None and hrv < 40 and ptype == "pain_flare":
        score += 0.05
        factors.append(ContributingFactor("low_hrv", 0.05, f"HRV {hrv:g}ms (low)"))

    if ptype == "migraine":
        drop = _num((weather or {}).get("pressure_change"))
        if drop is not None and drop < -8:
            score += 0.2
            factors.append(ContributingFactor("pressure_drop", 0.2, f"Pressure {drop:.1f} hPa/24h (drop)"))
        pollen = (weather or {}).get("pollen_level")
        if pollen in ("high", "very_high"):
            score += 0.15
            factors.append(ContributingFactor("high_pollen", 0.15, f"Pollen {pollen}"))

    return score


def _apply_body_pain(
    ptype: str,
    base: float,
    factors: list[ContributingFactor],
    body_pain: dict[str, Any] | None,
) -> float:
    avg_i = _num((body_pain or {}).get("avgIntensity"))
    if avg_i is None or avg_i <= 0:
        return base
    if ptype == "pain_flare":
        weight = (avg_i / 10) * 0.5
        factors.append(ContributingFactor("logged_body_pain", weight, f"Avg logged body pain {avg_i:.1f}/10"))
        return base + weight
    if ptype == "migraine" and avg_i >= 5:
        factors.append(ContributingFactor("logged_body_pain", 0.05, f"Avg logged body pain {avg_i:.1f}/10"))
        return base + 0.05
    return base


def _recurrence_for(ptype: str, entries: list[Entry]) -> int:
    window = entries[:7]
    if ptype == "migraine":
        return sum(1 for e in window if e.get("migraine") is True)
    if ptype == "anxiety":
        return sum(1 for e in window if (_num(e.get("anxiety")) or 0) >= 7)
    if ptype == "mood":
        return sum(
            1 for e in window if (_num(e.get("mood_valence")) or 0) < 0 or (_num(e.get("depression")) or 0) >= 6
        )
    if ptype == "pain_flare":
        return sum(1 for e in window if (_num(e.get("migraine_intensity")) or 0) >= 6)
    raise ValueError(f"unknown prediction_type {ptype!r}")


def _compute_one(ptype: str, inp: dict[str, Any]) -> dict[str, Any]:
    entries: list[Entry] = inp["entries"]
    factors: list[ContributingFactor] = []
    score = _base_for(ptype, entries, factors)
    score = _apply_wearable_weather(ptype, score, factors, inp.get("wearable"), inp.get("weather"))
    score = _apply_body_pain(ptype, score, factors, inp.get("bodyPain"))
    score = _clamp01(score)

    data_days = min(len(entries), 14)
    confidence = 0.4 + (data_days / 14) * 0.3
    recur = _recurrence_for(ptype, entries)
    if recur >= 3:
        confidence += 0.12
        factors.append(ContributingFactor("recurrence", 0, f"Pattern recurred {recur}× in 7 days"))
    wearable = inp.get("wearable")
    if wearable and (_num(wearable.get("hrv")) is not None or _num(wearable.get("sleep_score")) is not None):
        confidence += 0.1
    body_pain = inp.get("bodyPain")
    if ptype == "pain_flare" and body_pain and (_num(body_pain.get("daysWithPain")) or 0) >= 3:
        confidence += 0.1
        factors.append(
            ContributingFactor("body_pain_recurrence", 0, f"Pain logged on {body_pain.get('daysWithPain')} of last 7 days")
        )
    confidence = _clamp01(confidence)

    return {
        "prediction_type": ptype,
        "risk_score": round(score * 1000) / 1000,
        "risk_level": _level_for(score),
        "confidence": round(confidence * 1000) / 1000,
        "contributing_factors": factors,
        "model_version": BASELINE_MODEL_VERSION,
    }


def compute_predictions(inp: dict[str, Any]) -> list[dict[str, Any]]:
    """Port of ``computePredictions`` — risk for all four types, or ``[]`` when
    there are no entries. ``inp`` keys: entries (newest-first), wearable,
    weather, bodyPain."""
    if not inp.get("entries"):
        return []
    return [_compute_one(t, inp) for t in PREDICTION_TYPES]


# --------------------------------------------------------------------------- #
# Uniform model wrapper
# --------------------------------------------------------------------------- #
class RuleBaselineModel:
    """The rule engine as a :class:`Model`. Applies the abstention contract by
    default; Phase-5 fallback can disable abstention so rules always emit."""

    def __init__(self, *, apply_abstention: bool = True, min_history_days: int = MIN_HISTORY_DAYS):
        self.model_version = BASELINE_MODEL_VERSION
        self.apply_abstention = apply_abstention
        self.min_history_days = min_history_days

    def predict(self, window: PredictionWindow) -> list[Prediction]:
        inp = {
            "entries": window.entries,
            "wearable": window.wearable,
            "weather": window.weather,
            "bodyPain": window.body_pain,
        }
        comps = compute_predictions(inp)
        preds: list[Prediction] = []
        for c in comps:
            conf = c["confidence"]
            decision = (
                decide_abstention(
                    history_days=window.history_days,
                    confidence=conf,
                    min_history_days=self.min_history_days,
                )
                if self.apply_abstention
                else AbstentionDecision(False)
            )
            if decision.abstain:
                preds.append(
                    Prediction(
                        prediction_type=c["prediction_type"],
                        risk=None,
                        risk_level=None,
                        confidence=conf,
                        uncertainty=round(1 - conf, 3),
                        contributing_factors=c["contributing_factors"],
                        abstained=True,
                        model_version=self.model_version,
                        abstain_reason=decision.reason,
                    )
                )
            else:
                preds.append(
                    Prediction(
                        prediction_type=c["prediction_type"],
                        risk=c["risk_score"],
                        risk_level=c["risk_level"],
                        confidence=conf,
                        uncertainty=round(1 - conf, 3),
                        contributing_factors=c["contributing_factors"],
                        abstained=False,
                        model_version=self.model_version,
                    )
                )
        return preds
