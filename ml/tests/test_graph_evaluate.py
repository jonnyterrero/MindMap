"""Verifier eval harness + the adversarial LLM grounder.

The harness exists to MEASURE the verifier — chiefly false-accept rate. These
tests lock in what the conservative lexical grounder does: reliably blocks
hallucination/contradiction/psychological over-interpretation, surfaces supported
claims, and (documented limitation) false-accepts metaphor — motivating the real
NLI/LLM grounder.
"""

import json

from mindmap_ml.graph.evaluate import evaluate
from mindmap_ml.graph.ingest import digest
from mindmap_ml.graph.schema import CandidateGraph, Node
from mindmap_ml.graph.verify import (
    AnthropicGrounder,
    LexicalEntailment,
    LLMEntailment,
    make_entailment,
    verify_graph,
)


# --------------------------- eval harness (lexical) ------------------------- #
def test_harness_blocks_unsupported_categories() -> None:
    r = evaluate()
    # zero false-accepts where lexical overlap / negation is decisive
    assert r.per_category["hallucination"]["fa"] == 0
    assert r.per_category["contradiction"]["fa"] == 0
    assert r.per_category["psychological"]["fa"] == 0


def test_harness_surfaces_supported_claims() -> None:
    r = evaluate()
    assert r.false_reject == 0  # every supported claim surfaced
    assert r.recall >= 0.9
    assert r.per_category["supported"]["ta"] >= 1 and r.per_category["fragment"]["ta"] == 3


def test_harness_quantifies_figurative_false_accepts() -> None:
    # documented limitation: shallow grounding can't tell figurative from literal
    r = evaluate()
    assert r.per_category["metaphor"]["fa"] >= 1
    assert r.per_category["sarcasm"]["fa"] >= 1
    assert 0.0 < r.false_accept_rate < 1.0  # there ARE measured false accepts


def test_harness_blocks_overinterpretation_and_low_context() -> None:
    r = evaluate()
    assert r.per_category["emotional"]["fa"] == 0  # intense language != severe diagnosis
    assert r.per_category["low_context"]["fa"] == 0  # "ugh" != existential despair
    assert r.per_category["contradiction"]["tr"] >= 2  # both negated claims blocked


def test_harness_reports_calibration_and_serializes() -> None:
    r = evaluate()
    assert 0.0 <= r.brier <= 1.0 and 0.0 <= r.ece <= 1.0
    json.dumps(r.to_dict())  # report is serializable


# --------------------------- adversarial LLM grounder ----------------------- #
class FakeJSON:
    def __init__(self, resp: str) -> None:
        self.resp = resp
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.resp


def test_llm_entailment_parses_labels() -> None:
    assert LLMEntailment(FakeJSON('{"label":"entail","score":0.9}')).classify("p", "h") == ("entail", 0.9)
    assert LLMEntailment(FakeJSON('{"label":"contradict","score":0.7}')).classify("p", "h") == ("contradict", 0.7)
    # tolerates surrounding prose
    assert LLMEntailment(FakeJSON('sure: {"label":"neutral","score":0.2} done')).classify("p", "h")[0] == "neutral"


def test_llm_entailment_fails_closed() -> None:
    assert LLMEntailment(FakeJSON("not json")).classify("p", "h") == ("neutral", 0.0)
    assert LLMEntailment(FakeJSON('{"label":"bogus"}')).classify("p", "h") == ("neutral", 0.0)
    assert LLMEntailment(FakeJSON('{"label":"entail","score":1}')).classify("p", "") == ("neutral", 0.0)


def test_llm_grounder_plugs_into_verifier() -> None:
    doc, spans = digest("Work has been stressful.", user_id="u1")
    node = Node("n1", "stressful workload", "theme", evidence=[spans[0].span_id])
    # adversarial grounder says entail -> accepted
    art = verify_graph(doc, spans, CandidateGraph(doc.doc_id, [node], []),
                       entailment=LLMEntailment(FakeJSON('{"label":"entail","score":0.95}')))
    assert len(art.nodes) == 1 and art.nodes[0].status == "verified"
    assert art.verifier_versions["entailment"].startswith("llm_adversarial_v0")
    # adversarial grounder says neutral -> not surfaced (fail-closed)
    art2 = verify_graph(doc, spans, CandidateGraph(doc.doc_id, [Node("n1", "stressful workload", "theme", evidence=[spans[0].span_id])], []),
                        entailment=LLMEntailment(FakeJSON('{"label":"neutral","score":0.1}')))
    assert art2.nodes == []


# -------------------- make_entailment factory + AnthropicGrounder ----------- #
def test_make_entailment_returns_lexical_without_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ent = make_entailment(prefer_llm=True)
    assert isinstance(ent, LexicalEntailment)


def test_make_entailment_returns_llm_with_key(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    ent = make_entailment(prefer_llm=True)
    assert isinstance(ent, LLMEntailment)
    assert ent.version.startswith("llm_adversarial_v0")


def test_make_entailment_respects_prefer_llm_false(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    ent = make_entailment(prefer_llm=False)
    assert isinstance(ent, LexicalEntailment)


def test_anthropic_grounder_version_string() -> None:
    g = AnthropicGrounder()
    assert g.model == "claude-sonnet-4-6"
    g2 = AnthropicGrounder(model="claude-haiku-4-5-20251001")
    assert g2.model == "claude-haiku-4-5-20251001"
