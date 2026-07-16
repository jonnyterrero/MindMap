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
class GoldEdge:
    """Edge-level gold: src/dst index into the case's claims tuple."""

    src: int
    dst: int
    edge_type: str
    evidence_contains: tuple[str, ...]  # span locators to cite
    supported: bool  # should this edge surface (at any claim class)?
    category: str


@dataclass(frozen=True)
class GoldCase:
    case_id: str
    text: str
    claims: tuple[GoldClaim, ...]
    edges: tuple[GoldEdge, ...] = ()


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
    GoldCase(
        "sarcasm",
        "Oh sure, I just love being ignored.",
        # figurative/sarcastic: lexical overlap is high, so a shallow grounder is
        # confidently WRONG here (documented limitation -> motivates real NLI).
        (GoldClaim("love being ignored", "emotion", "love being ignored", False, "sarcasm"),),
    ),
    GoldCase(
        "emotional",
        "I am so stressed I could scream.",
        (
            GoldClaim("feeling stressed", "emotion", "so stressed", True, "supported"),
            # over-interpretation of intense language — must NOT surface
            GoldClaim("severe clinical depression", "value", "could scream", False, "emotional"),
        ),
    ),
    GoldCase(
        "low_context",
        "ugh.",
        (GoldClaim("deep existential despair", "emotion", "ugh", False, "low_context"),),
    ),
    GoldCase(
        "contradiction_negated",
        "I don't feel sad anymore.",
        (GoldClaim("feeling sad", "emotion", "feel sad", False, "contradiction"),),
    ),
    # ------------------------- clinical claims ------------------------------ #
    GoldCase(
        "clinical_reported",
        "I was diagnosed with migraine last year.",
        (
            # user's own explicit report — surfaces, but capped (never asserted
            # as directly_supported: the pipeline reports, it does not diagnose)
            GoldClaim("migraine diagnosis", "value", "diagnosed with migraine", True, "clinical"),
        ),
    ),
    GoldCase(
        "clinical_hedged",
        "My therapist said I might have ADHD.",
        # lexical overlap is total ("adhd" appears verbatim), so a shallow
        # grounder confidently ACCEPTS this — the clinical gate must catch the
        # hedge: "might have ADHD" does not license the claim "has ADHD".
        (GoldClaim("has ADHD", "value", "might have ADHD", False, "psychological"),),
    ),
    GoldCase(
        "clinical_inferred",
        "I keep losing focus at work.",
        # diagnosis invented from a behavior — must never surface
        (GoldClaim("ADHD", "value", "losing focus", False, "psychological"),),
    ),
    # ------------------------- causal edges --------------------------------- #
    GoldCase(
        "causal_explicit",
        # "only slept", not "barely slept": both nodes cite this one sentence,
        # and lexical_v0's polarity check reads "barely" as negation, which
        # would spuriously contradict "migraine came back".
        "My migraine came back because I only slept three hours.",
        (
            GoldClaim("slept three hours", "event", "slept three hours", True, "supported"),
            GoldClaim("migraine came back", "event", "migraine came back", True, "supported"),
        ),
        edges=(
            # explicit causal language in the text — directly supported
            GoldEdge(0, 1, "causal", ("because I only slept",), True, "causal_explicit"),
        ),
    ),
    GoldCase(
        "causal_grounded",
        "I barely slept. My migraine came back.",
        (
            GoldClaim("barely slept", "event", "barely slept", True, "supported"),
            GoldClaim("migraine came back", "event", "migraine came back", True, "supported"),
        ),
        edges=(
            # no causal language, but sleep->migraine maps to a curated prior:
            # surfaces as a weakly-inferred, cited hypothesis
            GoldEdge(0, 1, "causal", ("barely slept", "migraine came back"), True, "causal_grounded"),
        ),
    ),
    GoldCase(
        "causal_ungrounded",
        "I ate cereal this morning. My migraine came back.",
        (
            GoldClaim("ate cereal", "event", "ate cereal", True, "supported"),
            GoldClaim("migraine came back", "event", "migraine came back", True, "supported"),
        ),
        edges=(
            # no causal language AND no prior for cereal->migraine — an invented
            # cause that must NOT surface
            GoldEdge(0, 1, "causal", ("ate cereal", "migraine came back"), False, "causal_ungrounded"),
        ),
    ),
)
