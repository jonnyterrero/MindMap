"""Evidence scorer: clinical gate + causal-prior grounding (fail-closed)."""

from __future__ import annotations

from mindmap_ml.graph.evidence_scorer import (
    CausalGrounding,
    assertion_gate,
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
    assert any(m == "sleep_deficit->migraine" for m in g.matched)


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


# --------------------------------------------------------------------------- #
# assertion_gate — irrealis / attribution mood
# --------------------------------------------------------------------------- #
def test_conditional_premise_blocked() -> None:
    g = assertion_gate("If I sleep well, I feel calm the next day.")
    assert not g.allowed and "non_asserted_irrealis" in g.reasons


def test_wish_and_future_blocked() -> None:
    assert not assertion_gate("I wish I had slept more last night.").allowed
    assert not assertion_gate("I am going to start therapy next month.").allowed
    assert not assertion_gate("I'll call my therapist tomorrow.").allowed


def test_third_party_attribution_blocked() -> None:
    g = assertion_gate("My mom thinks I am overreacting about work.")
    assert not g.allowed and "attributed_to_third_party" in g.reasons
    assert not assertion_gate("She said I seem anxious.").allowed


def test_plain_assertion_allowed() -> None:
    assert assertion_gate("I feel anxious about work and I slept badly.").allowed
    assert assertion_gate("I woke up early and went for a walk.").allowed


def test_first_person_report_not_attributed() -> None:
    # first-person subject before the speech verb -> the writer's own assertion
    assert assertion_gate("I told my friend I was sad.").allowed


def test_intense_feeling_not_over_blocked() -> None:
    # "could" is deliberately NOT an irrealis marker: it co-occurs with real
    # asserted feeling ("so stressed I could scream") and must not cost recall.
    assert assertion_gate("I am so stressed I could scream.").allowed


def test_hopeful_is_not_a_wish() -> None:
    # token-level matching: "hopeful" must not trip the "hope"/"wish" markers
    assert assertion_gate("hopeful").allowed
    assert assertion_gate("I feel hopeful today.").allowed
