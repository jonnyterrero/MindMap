import pytest

from mindmap_ml.evidence.curate import (
    EvidencePrior,
    PriorValidationError,
    find_prior,
    load_priors,
)
from mindmap_ml.evidence.index import EvidenceIndex
from mindmap_ml.evidence.ingest import ingest_corpus
from mindmap_ml.evidence.retrieve import evidence_for, retrieve


# ------------------------------- priors ------------------------------------ #
def test_seed_priors_load_and_validate() -> None:
    priors = load_priors()
    assert len(priors) >= 5
    assert all(p.citation for p in priors)  # no uncited priors


def test_prior_schema_rejects_bad_values() -> None:
    with pytest.raises(PriorValidationError):
        EvidencePrior("sleep_deficit", "migraine", direction="up", strength="strong", citation="x")
    with pytest.raises(PriorValidationError):
        EvidencePrior("sleep_deficit", "not_an_outcome", direction="+", strength="strong", citation="x")
    with pytest.raises(PriorValidationError):
        EvidencePrior("sleep_deficit", "migraine", direction="+", strength="strong", citation="")


def test_find_prior() -> None:
    assert find_prior("sleep_deficit", "migraine") is not None
    assert find_prior("nonexistent_factor", "migraine") is None


# ------------------------------- ingest ------------------------------------ #
def test_ingest_seed_corpus_has_citations() -> None:
    passages = ingest_corpus()
    assert len(passages) >= 3
    assert all(p.citation and p.text for p in passages)
    sources = {p.source for p in passages}
    assert "sleep" in sources and "migraine_triggers" in sources


# ------------------------------- retrieval --------------------------------- #
def test_index_ranks_relevant_passage_first() -> None:
    idx = EvidenceIndex(ingest_corpus())
    top = idx.query("barometric pressure migraine attack", k=1)
    assert top and top[0].passage.source == "migraine_triggers"
    assert top[0].score > 0


def test_retrieve_returns_citations() -> None:
    results = retrieve("sleep and anxiety", k=2)
    assert results
    assert all(r.passage.citation for r in results)


def test_evidence_for_grounded_recommendation() -> None:
    ev = evidence_for("sleep_deficit", "migraine")
    assert ev.is_grounded
    assert ev.prior is not None
    assert ev.citations  # at least the prior's citation


def test_evidence_for_ungrounded_factor_has_no_prior() -> None:
    ev = evidence_for("totally_unknown_factor", "pain_flare")
    assert ev.prior is None
