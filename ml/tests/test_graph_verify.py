"""Stage 3 — external verification. The two load-bearing properties:
  * agreement != validation (a high-confidence hallucination is NOT returned)
  * fail-closed (unsupported causal/contradictory/anonymous claims blocked)
"""

from mindmap_ml.graph.ingest import digest
from mindmap_ml.graph.schema import CandidateGraph, Edge, Node
from mindmap_ml.graph.verify import verify_graph


def _verify(text: str, nodes: list[Node], edges: list[Edge] | None = None):
    doc, spans = digest(text, user_id="u1")
    cand = CandidateGraph(doc_id=doc.doc_id, nodes=nodes, edges=edges or [])
    return verify_graph(doc, spans, cand), spans


def test_span_supported_node_accepted() -> None:
    doc, spans = digest("I feel anxious about work.", user_id="u1")
    n = Node("n1", "anxious about work", "emotion", evidence=[spans[0].span_id])
    art = verify_graph(doc, spans, CandidateGraph(doc.doc_id, [n], []))
    assert len(art.nodes) == 1
    assert art.nodes[0].status == "verified"
    assert art.nodes[0].claim_class == "directly_supported"
    assert art.nodes[0].confidence and art.nodes[0].confidence.bucket == "high"


def test_hallucinated_node_not_returned_despite_high_confidence() -> None:
    # AGREEMENT != VALIDATION: generator is 0.99 sure but the span doesn't support it.
    doc, spans = digest("I had a productive day at work.", user_id="u1")
    n = Node("n1", "secret romantic affair", "entity", evidence=[spans[0].span_id], generator_confidence=0.99)
    art = verify_graph(doc, spans, CandidateGraph(doc.doc_id, [n], []))
    assert art.nodes == []  # not surfaced
    assert any(s["claim_id"] == "n1" and s["decision"] == "abstain" for s in art.suppressed)


def test_no_provenance_rejected() -> None:
    art, _ = _verify("anything here.", [Node("n1", "invented theme", "theme", evidence=[])])
    assert art.nodes == []
    assert any(s["claim_id"] == "n1" and "no_provenance" in s["reason_codes"] for s in art.suppressed)


def test_contradiction_rejected_and_escalated() -> None:
    doc, spans = digest("I am not calm at all today.", user_id="u1")
    n = Node("n1", "feeling calm", "emotion", evidence=[spans[0].span_id])
    art = verify_graph(doc, spans, CandidateGraph(doc.doc_id, [n], []))
    assert art.nodes == []
    d = next(d for d in art.decisions if d.claim_id == "n1")
    assert d.decision == "reject" and "nli_contradict" in d.reason_codes and d.escalated


def test_weakly_inferred_downgraded_not_promoted() -> None:
    # partial overlap (>= TAU_LOW) but not full -> downgrade, never directly_supported
    doc, spans = digest("Work has been really stressful and busy.", user_id="u1")
    n = Node("n1", "stressful workload", "theme", evidence=[spans[0].span_id],
             claim_class="weakly_inferred", inference_type="semantic")
    art = verify_graph(doc, spans, CandidateGraph(doc.doc_id, [n], []))
    assert len(art.nodes) == 1
    assert art.nodes[0].status == "downgraded" and art.nodes[0].claim_class == "weakly_inferred"


def test_invalid_node_type_rejected() -> None:
    art, _ = _verify("hello world.", [Node("n1", "x", "not_a_type", evidence=[])])
    assert art.nodes == []
    assert any("invalid_node_type" in s["reason_codes"] for s in art.suppressed)


def test_causal_edge_with_language_accepted() -> None:
    doc, spans = digest("I was anxious because I slept badly.", user_id="u1")
    a = Node("a", "anxious", "emotion", evidence=[spans[0].span_id])
    b = Node("b", "slept badly", "event", evidence=[spans[0].span_id])
    e = Edge("e1", "a", "b", "causal", evidence=[spans[0].span_id])
    art = verify_graph(doc, spans, CandidateGraph(doc.doc_id, [a, b], [e]))
    assert len(art.edges) == 1
    assert art.edges[0].status == "verified" and art.edges[0].claim_class == "directly_supported"


def test_causal_edge_without_language_downgraded() -> None:
    doc, spans = digest("I was anxious. I slept badly.", user_id="u1")
    a = Node("a", "anxious", "emotion", evidence=[spans[0].span_id])
    b = Node("b", "slept badly", "event", evidence=[spans[1].span_id])
    e = Edge("e1", "a", "b", "causal", evidence=[spans[0].span_id, spans[1].span_id])
    art = verify_graph(doc, spans, CandidateGraph(doc.doc_id, [a, b], [e]))
    assert len(art.edges) == 1
    assert art.edges[0].status == "downgraded"
    d = next(d for d in art.decisions if d.claim_id == "e1")
    assert "no_causal_language" in d.reason_codes


def test_dangling_edge_rejected_when_endpoint_not_verified() -> None:
    doc, spans = digest("I had a productive day at work.", user_id="u1")
    good = Node("a", "productive day at work", "event", evidence=[spans[0].span_id])
    halluc = Node("b", "secret affair", "entity", evidence=[spans[0].span_id])  # will abstain
    e = Edge("e1", "a", "b", "thematic", evidence=[spans[0].span_id])
    art = verify_graph(doc, spans, CandidateGraph(doc.doc_id, [good, halluc], [e]))
    assert {n.node_id for n in art.nodes} == {"a"}
    assert art.edges == []
    assert any(s["claim_id"] == "e1" and "dangling_edge" in s["reason_codes"] for s in art.suppressed)


def test_abstained_flag_when_nothing_survives() -> None:
    doc, spans = digest("I had a productive day at work.", user_id="u1")
    n = Node("n1", "alien invasion conspiracy", "theme", evidence=[spans[0].span_id])
    art = verify_graph(doc, spans, CandidateGraph(doc.doc_id, [n], []))
    assert art.abstained is True and art.nodes == []
    assert art.coverage["spans_total"] == 1
