"""The uniform model interface.

Every model — the rule baseline (Phase 1), forecasting models (Phase 3) — is a
callable that turns a :class:`PredictionWindow` into a list of
:class:`Prediction`. The eval harness consumes any object satisfying
:class:`Model`, so models are swappable and comparable on equal footing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable

PREDICTION_TYPES: tuple[str, ...] = ("migraine", "anxiety", "mood", "pain_flare")
RISK_LEVELS: tuple[str, ...] = ("low", "moderate", "high", "critical")


@dataclass
class ContributingFactor:
    """A signed, explainable contribution to a risk score (mirrors the TS
    ``ContributingFactor`` so persisted rows stay wire-compatible)."""

    factor: str
    weight: float  # signed contribution to the 0..1 score
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"factor": self.factor, "weight": self.weight}
        if self.detail is not None:
            d["detail"] = self.detail
        return d


@dataclass
class PredictionWindow:
    """Input to a model for one (user, as-of date) scoring point.

    ``entries`` are raw daily rows **newest-first** (matching the TS engines, so
    the ported baseline reproduces app output exactly). ``features`` is the
    engineered row ML models consume. Both views are provided so rule and ML
    models share one interface.
    """

    user_id: str
    as_of_date: date
    entries: list[dict[str, Any]]  # newest-first
    features: dict[str, float] | None = None
    wearable: dict[str, Any] | None = None
    weather: dict[str, Any] | None = None
    body_pain: dict[str, Any] | None = None

    @property
    def history_days(self) -> int:
        return len(self.entries)


@dataclass
class Prediction:
    """A single, gate-ready risk output.

    When ``abstained`` is True, ``risk``/``risk_level`` are ``None`` — the model
    declined to estimate (insufficient data / low confidence / OOD). Abstention
    is a feature, not a failure.
    """

    prediction_type: str
    risk: float | None  # 0..1, None when abstained
    risk_level: str | None  # low|moderate|high|critical, None when abstained
    confidence: float  # 0..1
    uncertainty: float  # 0..1, surfaced per the safety contract (≈ 1 - confidence)
    contributing_factors: list[ContributingFactor] = field(default_factory=list)
    abstained: bool = False
    model_version: str = "unknown"
    abstain_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_type": self.prediction_type,
            "risk": self.risk,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "contributing_factors": [f.to_dict() for f in self.contributing_factors],
            "abstained": self.abstained,
            "model_version": self.model_version,
            "abstain_reason": self.abstain_reason,
        }


@runtime_checkable
class Model(Protocol):
    """Anything the harness can evaluate."""

    model_version: str

    def predict(self, window: PredictionWindow) -> list[Prediction]: ...
