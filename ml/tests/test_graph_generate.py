"""Stage 2 — candidate generation: LLM parse, rule-skeleton fallback, dedup."""

import json

from mindmap_ml.graph.generate import generate_candidates
from mindmap_ml.graph.ingest import digest


class FakeExtractor:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    def generate_json(self, system: str, user: str) -> str:
        self.calls += 1
        return self.payload


def test_rule_skeleton_when_no_client() -> None:
    doc, spans = digest("I feel anxious. Work is hard.", user_id="u1")
    g = generate_candidates(doc, spans, client=None)
    assert g.method == "rule_skeleton"
    assert len(g.nodes) == 2
    assert g.edges == []
    for n, sp in zip(g.nodes, spans, strict=True):
        assert n.evidence == [sp.span_id]
        assert n.claim_class == "directly_supported"


def test_llm_parse_basic() -> None:
    doc, spans = digest("I was anxious because I slept badly.", user_id="u1")
    payload = json.dumps({
        "nodes": [
            {"id": "a", "label": "anxious", "type": "emotion", "evidence": [spans[0].span_id], "confidence": 0.8},
            {"id": "b", "label": "slept badly", "type": "event", "evidence": [spans[0].span_id], "confidence": 0.7},
        ],
        "edges": [
            {"src": "a", "dst": "b", "type": "causal", "evidence": [spans[0].span_id], "confidence": 0.6},
        ],
    })
    fake = FakeExtractor(payload)
    g = generate_candidates(doc, spans, client=fake)
    assert fake.calls == 1 and g.method == "llm"
    assert len(g.nodes) == 2 and len(g.edges) == 1
    assert g.edges[0].src == g.nodes[0].node_id and g.edges[0].dst == g.nodes[1].node_id


def test_invalid_spans_and_dangling_edges_dropped() -> None:
    doc, spans = digest("Work is hard.", user_id="u1")
    payload = json.dumps({
        "nodes": [{"id": "a", "label": "work", "type": "theme", "evidence": [spans[0].span_id, "sp_bogus"]}],
        "edges": [{"src": "a", "dst": "ghost", "type": "thematic", "evidence": []}],
    })
    g = generate_candidates(doc, spans, client=FakeExtractor(payload), dedup=False)
    assert g.nodes[0].evidence == [spans[0].span_id]  # bogus span dropped
    assert g.edges == []  # edge to unknown node dropped


def test_malformed_json_falls_back_to_skeleton() -> None:
    doc, spans = digest("I feel tired.", user_id="u1")
    g = generate_candidates(doc, spans, client=FakeExtractor("not json at all"))
    assert g.method == "rule_skeleton"
    assert len(g.nodes) == 1


def test_dedup_merges_and_remaps_edges() -> None:
    doc, spans = digest("Work stress. Work stress again. Sleep.", user_id="u1")
    payload = json.dumps({
        "nodes": [
            {"id": "n1", "label": "work stress", "type": "theme", "evidence": [spans[0].span_id]},
            {"id": "n2", "label": "work stress", "type": "theme", "evidence": [spans[1].span_id]},
            {"id": "n3", "label": "sleep", "type": "theme", "evidence": [spans[2].span_id]},
        ],
        "edges": [{"src": "n2", "dst": "n3", "type": "thematic", "evidence": [spans[2].span_id]}],
    })
    g = generate_candidates(doc, spans, client=FakeExtractor(payload), dedup=True)
    assert len(g.nodes) == 2  # n1 & n2 merged
    merged = g.nodes[0]
    assert set(merged.evidence) == {spans[0].span_id, spans[1].span_id}
    # edge n2->n3 remapped to n1->n3 and preserved
    assert len(g.edges) == 1 and g.edges[0].src == merged.node_id


def test_empty_spans_returns_empty_graph() -> None:
    doc, spans = digest("   ", user_id="u1")
    g = generate_candidates(doc, spans, client=None)
    assert g.nodes == [] and g.edges == []
