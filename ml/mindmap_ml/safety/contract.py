"""The abstention contract.

Rule #1 of the safety contract: when data is insufficient or model uncertainty
is high, **abstain** rather than emit a number. This module is the single place
that decides abstention so every model and the serving job agree.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import (
    INSUFFICIENT_DATA_MESSAGE,
    MIN_CONFIDENCE_TO_SURFACE,
    MIN_HISTORY_DAYS,
)

CONFIDENCE_LEVELS = ("low", "medium", "high")


def confidence_level(confidence: float) -> str:
    """Bucket a 0..1 confidence into low/medium/high."""
    if confidence >= 0.66:
        return "high"
    if confidence >= 0.4:
        return "medium"
    return "low"


@dataclass(frozen=True)
class Uncertainty:
    """Calibrated uncertainty that ships with every prediction."""

    confidence: float  # 0..1
    level: str  # low|medium|high

    @classmethod
    def from_confidence(cls, confidence: float) -> Uncertainty:
        c = max(0.0, min(1.0, confidence))
        return cls(confidence=c, level=confidence_level(c))

    @property
    def value(self) -> float:
        """Uncertainty as 1 - confidence."""
        return round(1.0 - self.confidence, 6)


@dataclass(frozen=True)
class AbstentionDecision:
    abstain: bool
    reason: str | None = None
    user_message: str | None = None


def decide_abstention(
    *,
    history_days: int,
    confidence: float,
    missing_key_features: bool = False,
    out_of_distribution: bool = False,
    model_available: bool = True,
    calibration_ok: bool = True,
    min_history_days: int = MIN_HISTORY_DAYS,
    min_confidence: float = MIN_CONFIDENCE_TO_SURFACE,
) -> AbstentionDecision:
    """Return whether to abstain, in priority order. Any True → abstain.

    Reasons are stable strings (for telemetry/tests); the user-facing message is
    deliberately gentle and identical for the data-thin cases.
    """
    if not model_available:
        return AbstentionDecision(True, "model_unavailable", INSUFFICIENT_DATA_MESSAGE)
    if history_days < min_history_days:
        return AbstentionDecision(True, "insufficient_history", INSUFFICIENT_DATA_MESSAGE)
    if missing_key_features:
        return AbstentionDecision(True, "missing_key_features", INSUFFICIENT_DATA_MESSAGE)
    if out_of_distribution:
        return AbstentionDecision(True, "out_of_distribution", INSUFFICIENT_DATA_MESSAGE)
    if not calibration_ok:
        return AbstentionDecision(True, "calibration_failed", INSUFFICIENT_DATA_MESSAGE)
    if confidence < min_confidence:
        return AbstentionDecision(True, "low_confidence", INSUFFICIENT_DATA_MESSAGE)
    return AbstentionDecision(False, None, None)
