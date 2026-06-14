"""Clinician-shareable summary — the integrative Tier-0 deliverable.

Ties together everything that's honest at n-of-1: data completeness + readiness
(from the power analysis), metric trajectories, evidence-grounded conditional
patterns + lagged correlations, naive next-day/next-week watch items, optional
PHQ-9/GAD-7 screening, and safety flags. Abstains below a hard data floor, runs
every user-facing string through the output gate, and frames everything as
patterns, never diagnoses.

Single-user input. Pure (no I/O); the batch/serving layer persists the result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from ..evidence.retrieve import evidence_for
from ..features.calendar import logging_stats
from ..insights.correlations import compute_lagged_correlations
from ..insights.descriptive import Condition, conditional_rate, trend
from ..insights.naive_forecast import next_event_probability
from ..labels.instruments import Gad7Result, Phq9Result
from ..safety.crisis import CRISIS_RESOURCES, crisis_header, detect_crisis
from ..safety.gate import check_output

DATE_COL = "entry_date"

# From synthetic/power: |r|>=0.3 needs ~30 logged days to be reliable (FP ~30% at 14d).
DEFAULT_MIN_DAYS = 30
HARD_MIN_DAYS = 7

KEY_METRICS = ("anxiety", "depression", "mood_valence", "sleep_minutes", "focus")

# (evidence_factor, evidence_outcome, trigger, outcome) — patterns we'll test, each
# grounded in a curated prior / retrieved passage before it may be surfaced.
_PATTERN_CANDIDATES: list[tuple[str, str, Condition, Condition]] = [
    ("sleep_deficit", "anxiety", ("sleep_minutes", "<", 360), ("anxiety", ">=", 7)),
    ("sleep_deficit", "migraine", ("sleep_minutes", "<", 360), ("migraine", "==", True)),
    ("sleep_deficit", "depression", ("sleep_minutes", "<", 360), ("depression", ">=", 6)),
]
_FORECAST_EVENTS: list[Condition] = [("anxiety", ">=", 7), ("depression", ">=", 6)]

_DISCLAIMERS = [
    "This is a pattern summary from self-tracked data — not a diagnosis or medical advice.",
    "Associations are possible patterns, not proven causes.",
    "Discuss anything concerning with a qualified professional.",
]


def _gate(text: str, *, is_risk_claim: bool = True) -> str:
    res = check_output(text, is_risk_claim=is_risk_claim)
    return res.text if res.allowed and res.text else (res.safe_text or text)


@dataclass
class GroundedPattern:
    statement: str
    citations: list[str]


@dataclass
class ClinicianSummary:
    user_id: str
    date_range: list[str]  # [start, end] isoformat
    abstained: bool
    completeness: dict[str, Any]
    readiness: dict[str, Any]
    trajectories: list[dict[str, Any]] = field(default_factory=list)
    detected_patterns: list[GroundedPattern] = field(default_factory=list)
    watch_items: list[dict[str, Any]] = field(default_factory=list)
    instruments: dict[str, Any] = field(default_factory=dict)
    safety_flags: list[str] = field(default_factory=list)
    crisis: dict[str, Any] | None = None
    disclaimers: list[str] = field(default_factory=lambda: list(_DISCLAIMERS))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["detected_patterns"] = [asdict(p) for p in self.detected_patterns]
        return d


def _safety(phq9: Phq9Result | None, notes_text: str | None) -> tuple[list[str], dict[str, Any] | None]:
    flags: list[str] = []
    severity = None
    if phq9 is not None and phq9.suicidality_flag:
        flags.append("phq9_item9_positive")
        severity = "critical"
    crisis_sev = detect_crisis(notes_text)
    if crisis_sev:
        flags.append(f"crisis_language:{crisis_sev}")
        severity = severity or crisis_sev
    if not severity:
        return flags, None
    header = crisis_header("critical" if severity == "critical" else crisis_sev or "concern")
    return flags, {
        "severity": severity,
        "title": header["title"],
        "body": header["body"],
        "resources": [{"label": r.label, "detail": r.detail, "href": r.href} for r in CRISIS_RESOURCES],
    }


def build_clinician_summary(
    df: pd.DataFrame,
    *,
    phq9: Phq9Result | None = None,
    gad7: Gad7Result | None = None,
    notes_text: str | None = None,
    min_days: int = DEFAULT_MIN_DAYS,
) -> ClinicianSummary:
    """Build a single user's clinician-shareable summary."""
    if df.empty:
        return ClinicianSummary(
            user_id="unknown", date_range=[], abstained=True,
            completeness={"logged_days": 0}, readiness={"ready": False, "recommended_min_days": min_days, "logged_days": 0},
        )

    stats = logging_stats(df)[0]
    dates = sorted(df[DATE_COL].tolist())
    date_range = [str(dates[0]), str(dates[-1])]
    completeness = {
        "logged_days": stats.logged_days,
        "span_days": stats.span_days,
        "adherence": stats.adherence,
        "current_streak": stats.current_streak,
        "longest_streak": stats.longest_streak,
    }
    readiness = {
        "logged_days": stats.logged_days,
        "recommended_min_days": min_days,
        "ready": stats.logged_days >= min_days,
        "days_remaining": max(0, min_days - stats.logged_days),
    }
    instruments: dict[str, Any] = {}
    if phq9 is not None:
        instruments["phq9"] = {"total": phq9.total, "severity": phq9.severity, "disclaimer": phq9.disclaimer}
    if gad7 is not None:
        instruments["gad7"] = {"total": gad7.total, "severity": gad7.severity, "disclaimer": gad7.disclaimer}

    safety_flags, crisis = _safety(phq9, notes_text)

    # Below the hard floor: abstain on patterns/forecasts, keep safety + readiness.
    if stats.logged_days < HARD_MIN_DAYS:
        return ClinicianSummary(
            user_id=stats.user_id, date_range=date_range, abstained=True,
            completeness=completeness, readiness=readiness, instruments=instruments,
            safety_flags=safety_flags, crisis=crisis,
        )

    trajectories: list[dict[str, Any]] = []
    for m in KEY_METRICS:
        t = trend(df, m)
        if t is not None:
            trajectories.append({
                "metric": t.metric, "label": t.label, "direction": t.direction,
                "mean": t.mean, "statement": _gate(t.statement, is_risk_claim=False),
            })

    detected: list[GroundedPattern] = []
    for factor, outcome_name, trig, out in _PATTERN_CANDIDATES:
        cr = conditional_rate(df, trig, out)
        if cr is None:
            continue
        meaningful = cr.lift is None or cr.lift >= 1.2 or cr.rate > cr.baseline
        if not meaningful:
            continue
        ev = evidence_for(factor, outcome_name)
        if not ev.is_grounded:  # no citation -> do not surface a recommendation-grade claim
            continue
        detected.append(GroundedPattern(_gate(cr.statement, is_risk_claim=True), ev.citations))

    # Lagged correlations describe the user's OWN data (not advice) -> no citation required.
    for c in compute_lagged_correlations(df)[:3]:
        detected.append(GroundedPattern(_gate(c.statement, is_risk_claim=True), []))

    watch: list[dict[str, Any]] = []
    for out in _FORECAST_EVENTS:
        for h in (1, 7):
            f = next_event_probability(df, out, horizon=h)
            if not f.abstained and f.probability is not None:
                watch.append({
                    "outcome": out[0], "horizon": h, "probability": f.probability,
                    "method": f.method, "statement": _gate(f.statement, is_risk_claim=True),
                })

    return ClinicianSummary(
        user_id=stats.user_id, date_range=date_range, abstained=False,
        completeness=completeness, readiness=readiness, trajectories=trajectories,
        detected_patterns=detected, watch_items=watch, instruments=instruments,
        safety_flags=safety_flags, crisis=crisis,
    )
