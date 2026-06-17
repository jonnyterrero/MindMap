"""End-to-end: raw text -> verified MindmapArtifact, with provenance traceable
back to real spans (offset-valid) and hallucinations suppressed."""

import json

from mindmap_ml.graph.ingest import digest
from mindmap_ml.graph.pipeline import run_pipeline


class FakeExtractor:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def generate_json(self, system: str, user: str) -> str:
        return self.payload


def test_end_to_end_rule_skeleton_is_traceable() -> None:
    text = "I feel anxious about work. I slept badly."
    art = run_pipeline(text, user_id="u1")  # no client -> rule skeleton
    assert not art.abstained
    assert len(art.nodes) >= 1

    doc, spans = digest(text, user_id="u1")  # deterministic -> same ids
    by_id = {sp.span_id: sp for sp in spans}
    for n in art.nodes:
        assert n.evidence, "every node must carry provenance"
        for sid in n.evidence:
            assert sid in by_id
            sp = by_id[sid]
            # provenance is highlightable in the original text
            assert doc.raw_text[sp.raw_start : sp.raw_end]
            assert n.confidence is not None
    assert art.coverage["spans_total"] == len(spans)
    assert art.verifier_versions["entailment"] == "lexical_v0"


def test_end_to_end_llm_suppresses_hallucination() -> None:
    text = "I had a productive day at work."
    doc, spans = digest(text, user_id="u1")
    payload = json.dumps({
        "nodes": [
            {"id": "a", "label": "productive day at work", "type": "event",
             "evidence": [spans[0].span_id], "confidence": 0.7},
            {"id": "b", "label": "secret affair", "type": "entity",
             "evidence": [spans[0].span_id], "confidence": 0.95},
        ],
        "edges": [],
    })
    art = run_pipeline(text, user_id="u1", extractor_client=FakeExtractor(payload))
    labels = {n.label for n in art.nodes}
    assert "productive day at work" in labels
    assert "secret affair" not in labels  # blocked despite 0.95 generator confidence
    assert any(s["decision"] == "abstain" for s in art.suppressed)


def test_end_to_end_is_deterministic() -> None:
    text = "Work stress. I slept badly. I feel hopeful."
    a = run_pipeline(text, user_id="u1")
    b = run_pipeline(text, user_id="u1")
    assert [n.label for n in a.nodes] == [n.label for n in b.nodes]
    assert a.coverage == b.coverage
    assert [s["claim_id"] for s in a.suppressed] == [s["claim_id"] for s in b.suppressed]
