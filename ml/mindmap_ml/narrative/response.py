"""The evidence-grounded response contract: ``{answer, sources, confidence}``.

This is the anti-hallucination boundary. A response is only allowed to contain a
model-written ``answer`` when the prediction is non-abstained AND the evidence is
grounded (a curated prior or a retrieved passage) AND the guardrails pass.
Otherwise it returns a fixed fallback and **never** fabricates:

- prediction abstained (not enough of the user's own data) → the gentle
  "not enough data yet" message, no sources.
- no grounding evidence in the corpus → ``"Insufficient evidence in training
  data."``, no sources, confidence 0 — the LLM is never even called.
- model text tripped the safety gate → the safe replacement, reduced confidence.

The LLM client is injected (so this is testable without a key/network).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import INSUFFICIENT_DATA_MESSAGE
from ..evidence.retrieve import Evidence
from ..models.base import Prediction
from .compose import LLMClient, compose_narrative

INSUFFICIENT_EVIDENCE = "Insufficient evidence in training data."
# A blocked answer is still safe, but we don't trust it — cap its confidence.
_BLOCKED_CONFIDENCE_CAP = 0.3


@dataclass
class GroundedResponse:
    answer: str
    sources: list[str]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"answer": self.answer, "sources": self.sources, "confidence": self.confidence}


def build_response(
    prediction: Prediction,
    evidence: Evidence,
    *,
    client: LLMClient | None = None,
) -> GroundedResponse:
    """Produce the ``{answer, sources, confidence}`` contract for one prediction.

    Fabrication is impossible: the LLM is only invoked on the grounded, non-abstained
    path, and even then its text must clear the guardrails or it is replaced.
    """
    conf = round(float(prediction.confidence), 3)

    # 1) Not enough of the USER's data → abstain (no sources, no LLM).
    if prediction.abstained:
        return GroundedResponse(INSUFFICIENT_DATA_MESSAGE, [], conf)

    # 2) No grounding evidence in the corpus → fixed fallback (no sources, no LLM).
    if not evidence.is_grounded:
        return GroundedResponse(INSUFFICIENT_EVIDENCE, [], 0.0)

    # 3) Grounded → compose + guard.
    narrative = compose_narrative(prediction, evidence, client=client)
    if narrative.blocked:
        return GroundedResponse(narrative.text, narrative.citations, min(conf, _BLOCKED_CONFIDENCE_CAP))
    return GroundedResponse(narrative.text, narrative.citations, conf)
