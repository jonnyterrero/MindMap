"""Retrieval-evidence scorer for interpretive claims (HANDOFF item #3).

Two claim families get an EXTRA, independent gate on top of entailment:

  * Clinical/diagnostic node claims ("anxiety disorder", "ADHD", ...) —
    a mindmap must never assert a diagnosis the user's own text doesn't name.
    Even when the text names it, the claim is CAPPED at weakly_inferred: the
    pipeline reports what the user said, it does not diagnose.
  * Causal edges without explicit causal language — an inferred cause may
    keep surfacing (as a cited hypothesis) only if the (factor, outcome)
    pair maps to a curated evidence prior; otherwise it is down-ranked out.

Safety invariant, enforced by construction: evidence grounding can only
DEMOTE or CAP a claim, never promote one. Literature plausibility must not
launder claims the text itself does not support. Everything here is
deterministic and auditable (frozen lexicons + the reviewed priors table) —
no model calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..evidence.curate import EvidencePrior, load_priors

SCORER_VERSION = "evidence_scorer_v1"

_WORD_RE = re.compile(r"[a-z][a-z']+")

# Diagnostic framing — terms that turn a description into a clinical assertion.
# Multi-word terms are matched as phrases in the lowercased label.
_CLINICAL_PHRASES: tuple[str, ...] = (
    "anxiety disorder",
    "panic disorder",
    "bipolar disorder",
    "eating disorder",
    "personality disorder",
    "clinical depression",
    "major depression",
    "depressive episode",
    "manic episode",
    "panic attack",
)
# Generic diagnostic framing vs named conditions: a label that only *frames*
# ("migraine diagnosis") is checked against its remaining content tokens,
# while a label naming a condition ("ADHD") requires that condition itself
# to appear in the cited text.
_GENERIC_FRAMING = frozenset(
    ["disorder", "diagnosis", "clinical", "syndrome", "disease", "condition"]
)
_CONDITION_TERMS = frozenset(
    [
        "adhd",
        "ocd",
        "ptsd",
        "bipolar",
        "depression",
        "insomnia",
        "mania",
        "psychosis",
        "schizophrenia",
        "autism",
    ]
)
_CLINICAL_TERMS = _GENERIC_FRAMING | _CONDITION_TERMS

# Hedges: text that merely wonders about a condition does not license a claim
# that asserts it ("might have ADHD" must not become "has ADHD").
_HEDGES = frozenset(
    ["might", "maybe", "perhaps", "possibly", "probably", "wonder", "wondering", "unsure", "suspect", "could"]
)

# Function words ignored when checking a framing-only label against the text.
_GATE_STOP = frozenset(
    ["the", "a", "an", "of", "with", "for", "and", "my", "your", "his", "her", "their", "has", "have", "had", "was", "is", "are", "been", "being", "severe", "mild", "moderate"]
)

# Tiny, auditable lemma table so surface forms meet the priors' vocabulary
# ("slept badly" must match factor sleep_deficit). Extend deliberately —
# every entry is a reviewable modeling decision, not a stemmer heuristic.
_LEMMAS: dict[str, str] = {
    "diagnosed": "diagnosis",
    "diagnoses": "diagnosis",
    "slept": "sleep",
    "sleeping": "sleep",
    "sleepless": "sleep",
    "anxious": "anxiety",
    "anxiousness": "anxiety",
    "depressed": "depression",
    "depressive": "depression",
    "migraines": "migraine",
    "manic": "mania",
    "moods": "mood",
}


def _norm_tokens(text: str) -> set[str]:
    return {_LEMMAS.get(t, t) for t in _WORD_RE.findall(text.lower())}


def is_clinical(label: str) -> bool:
    """Does this claim label assert a clinical/diagnostic construct?"""
    low = label.lower()
    if any(p in low for p in _CLINICAL_PHRASES):
        return True
    return bool(_norm_tokens(label) & _CLINICAL_TERMS)


@dataclass
class GateResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


def clinical_gate(label: str, premise: str) -> GateResult:
    """Gate a clinical node claim against the user's own cited text.

    Blocks when the cited text never names the condition (interpretive leap)
    or names it only under a hedge (the claim would overstate the text).
    When allowed, the caller must still cap the claim at weakly_inferred.
    """
    label_tokens = _norm_tokens(label)
    for phrase in _CLINICAL_PHRASES:
        if phrase in label.lower():
            label_tokens |= {_LEMMAS.get(t, t) for t in phrase.split()}
    premise_tokens = _norm_tokens(premise)

    conditions = label_tokens & _CONDITION_TERMS
    if conditions:
        # the named condition itself must appear in the user's text
        named_in_text = bool(conditions & premise_tokens)
    else:
        # framing-only label ("migraine diagnosis"): the substantive tokens
        # around the framing must appear in the text
        substantive = label_tokens - _GENERIC_FRAMING - _GATE_STOP
        named_in_text = bool(substantive) and substantive <= premise_tokens
    if not named_in_text:
        return GateResult(False, ["clinical_term_not_in_text"])
    if premise_tokens & _HEDGES:
        return GateResult(False, ["hedged_clinical_claim"])

    # Allowed (capped by caller). Attach any matching prior citations so the
    # surfaced claim carries its evidence context.
    citations = [
        p.citation
        for p in load_priors()
        if p.outcome in premise_tokens or p.outcome in label_tokens
    ]
    return GateResult(True, ["clinical_claim_capped"], _dedupe(citations))


@dataclass
class CausalGrounding:
    grounded: bool
    citations: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)  # "factor->outcome" pairs


def ground_causal(src_label: str, dst_label: str) -> CausalGrounding:
    """Map an inferred causal edge onto the curated priors table.

    Orientation-agnostic on purpose: the prior grounds the *pairing* of
    constructs (sleep <-> anxiety); the edge direction remains the user's
    weakly-inferred hypothesis either way. Prior-only — free-text retrieval
    is too easy to satisfy with one shared token to gate causality on.
    """
    src, dst = _norm_tokens(src_label), _norm_tokens(dst_label)
    citations: list[str] = []
    matched: list[str] = []
    for p in _prior_pairs():
        factor_hit_src = bool(p.factor_tokens & src)
        factor_hit_dst = bool(p.factor_tokens & dst)
        outcome_hit_src = p.prior.outcome in src
        outcome_hit_dst = p.prior.outcome in dst
        if (factor_hit_src and outcome_hit_dst) or (factor_hit_dst and outcome_hit_src):
            citations.append(p.prior.citation)
            matched.append(f"{p.prior.factor}->{p.prior.outcome}")
    return CausalGrounding(bool(matched), _dedupe(citations), matched)


@dataclass(frozen=True)
class _PriorPair:
    prior: EvidencePrior
    factor_tokens: frozenset[str]


def _prior_pairs() -> list[_PriorPair]:
    return [
        _PriorPair(p, frozenset(_norm_tokens(p.factor.replace("_", " "))))
        for p in load_priors()
    ]


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
