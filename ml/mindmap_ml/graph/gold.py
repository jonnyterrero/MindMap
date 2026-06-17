"""Hand-authored gold + challenge set for verifier evaluation.

Pre-launch there is no real labeled data, so this is human-authored ground truth
(clearly synthetic) covering the failure modes that matter: supported claims that
SHOULD surface, and hallucination / contradiction / metaphor / psychological
over-interpretation that should NOT. It is small on purpose — it exists to MEASURE
the verifier (false-accept rate is the headline), not to train it. Replace/expand
with ~200 real, dual-annotated entries before trusting absolute numbers.

Each GoldClaim is what a generator might propose; `supported` is the human verdict
on whether the cited text actually grounds it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldClaim:
    label: str
    node_type: str
    evidence_contains: str | None  # locate the span to cite (None -> no provenance)
    supported: bool  # human ground truth: does the text actually support this?
    category: str


@dataclass(frozen=True)
class GoldCase:
    case_id: str
    text: str
    claims: tuple[GoldClaim, ...]


GOLD_CASES: tuple[GoldCase, ...] = (
    GoldCase(
        "supported_basic",
        "I feel anxious about work and I slept badly.",
        (
            GoldClaim("anxious about work", "emotion", "anxious about work", True, "supported"),
            GoldClaim("slept badly", "event", "slept badly", True, "supported"),
        ),
    ),
    GoldCase(
        "hallucination",
        "I had a productive day at work.",
        (
            GoldClaim("productive day at work", "event", "productive day", True, "supported"),
            GoldClaim("secret affair", "entity", "productive day", False, "hallucination"),
        ),
    ),
    GoldCase(
        "contradiction",
        "I am not anxious at all today.",
        (GoldClaim("anxious", "emotion", "not anxious", False, "contradiction"),),
    ),
    GoldCase(
        "metaphor",
        "Work is a black hole swallowing all my time.",
        (GoldClaim("literal black hole", "entity", "black hole", False, "metaphor"),),
    ),
    GoldCase(
        "psychological",
        "I keep checking my phone every few minutes.",
        (
            GoldClaim("checking phone frequently", "event", "checking my phone", True, "supported"),
            GoldClaim("anxiety disorder", "value", "checking my phone", False, "psychological"),
        ),
    ),
    GoldCase(
        "fragments",
        "tired\nstressed\nhopeful",
        (
            GoldClaim("tired", "emotion", "tired", True, "fragment"),
            GoldClaim("stressed", "emotion", "stressed", True, "fragment"),
            GoldClaim("hopeful", "emotion", "hopeful", True, "fragment"),
        ),
    ),
)
