"""Evidence scorer: clinical gate + causal-prior grounding (fail-closed)."""

from __future__ import annotations

from mindmap_ml.graph.evidence_scorer import (
    CausalGrounding,
    clinical_gate,
    ground_causal,
    is_clinical,
)


# --------------------------------------------------------------------------- #
# is_clinical
# --------------------------------------------------------------------------- #
def test_diagnostic_labels_detected() -> None:
    assert is_clinical("anxiety disorder")
    assert is_clinical("ADHD")
    assert is_clinical("severe clinical depression")
    assert is_clinical("bipolar episode")


def test_ordinary_emotion_labels_not_clinical() -> None:
    # everyday emotion words must NOT trip the diagnostic gate
    assert not is_clinical("anxious about work")
    assert not is_clinical("feeling stressed")
    assert not is_clinical("hopeful")
    assert not is_clinical("slept badly")


# --------------------------------------------------------------------------- #
# clinical_gate
# --------------------------------------------------------------------------- #
def test_condition_never_named_in_text_is_blocked() -> None:
    g = clinical_gate("anxiety disorder", "I keep checking my phone every few minutes.")
    assert not g.allowed
    assert "clinical_term_not_in_text" in g.reasons


def test_hedged_mention_does_not_license_assertion() -> None:
    # "might have ADHD" must not become the claim "has ADHD"
    g = clinical_gate("has ADHD", "My therapist said I might have ADHD.")
    assert not g.allowed
    assert "hedged_clinical_claim" in g.reasons


def test_explicit_self_report_is_allowed_but_capped() -> None:
    g = clinical_gate("migraine diagnosis", "I was diagnosed with migraine last year.")
    assert g.allowed
    assert "clinical_claim_capped" in g.reasons


def test_gate_never_upgrades_only_blocks_or_caps() -> None:
    # every allowed outcome carries the cap reason — there is no plain pass
    g = clinical_gate("depression", "I was treated for depression.")
    assert g.allowed and "clinical_claim_capped" in g.reasons


# --------------------------------------------------------------------------- #
# ground_causal
# --------------------------------------------------------------------------- #
def test_prior_backed_pair_grounds_with_citation() -> None:
    g = ground_causal("barely slept", "migraine came back")
    assert g.grounded
    assert g.citations
    assert any("sleep_deficit->migraine" == m for m in g.matched)


def test_orientation_agnostic() -> None:
    # sleep_deficit -> anxiety prior grounds the pair in either direction
    fwd = ground_causal("slept badly", "anxious")
    rev = ground_causal("anxious", "slept badly")
    assert fwd.grounded and rev.grounded


def test_unmapped_pair_is_ungrounded() -> None:
    g = ground_causal("ate cereal", "migraine came back")
    assert g == CausalGrounding(False, [], [])


def test_lemma_normalization_meets_prior_vocabulary() -> None:
    # surface forms: slept/anxious vs priors' sleep_deficit/anxiety
    assert ground_causal("slept terribly", "felt anxious").grounded
