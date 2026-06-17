"""Stage 3 — External Verification (hybrid). The generator cannot self-approve.

This is a battery of INDEPENDENT checkers with an entailment objective — not a
second generator prompt. Agreement with the generator counts for nothing; only
independent evidence does. Components:

  (A) schema validator        — deterministic, fail-closed
  (B) provenance checker      — deterministic
  (C) entailment grounder     — SEPARATE model (injectable; default = a conservative
                                lexical placeholder. Replace with a cross-encoder NLI
                                (DeBERTa-MNLI) or an adversarial LLM verifier.)
  (D) graph consistency       — dangling edges, contradictions (flag, never merge)
  (E) calibrator              — rule-based confidence + final claim_class (trained
                                calibrator is future; needs a gold set)

Decision per claim: accept | downgrade | abstain | reject. Unsupported-but-plausible
structure does NOT pass; low-evidence is downgraded or omitted; schema/contradiction/
no-provenance fail closed.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Protocol

from .schema import (
    CLAIM_CLASSES,
    EDGE_TYPES,
    INFERENCE_TYPES,
    NODE_TYPES,
    CandidateGraph,
    Confidence,
    Edge,
    MindmapArtifact,
    Node,
    RawDocument,
    TextSpan,
    VerifierDecision,
)

TAU_LOW = 0.5  # min lexical overlap to treat a span as motivating an inferred claim
VERIFY_VERSION = "verify_v0"

_WORD_RE = re.compile(r"[a-z][a-z']+")
_STOP = frozenset(
    ["the", "a", "an", "and", "i", "to", "of", "it", "is", "in", "that", "you", "my", "for", "was", "on", "so", "but", "with", "this", "me", "we", "he", "she", "they", "am", "are", "be", "been", "being", "do", "did", "does", "have", "has", "had", "will", "would", "can", "could", "not", "no", "my", "your", "our"]
)
_NEG = frozenset(["not", "no", "never", "without", "cannot", "none", "nothing", "hardly", "barely"])
_CAUSAL = frozenset(
    ["because", "cause", "caused", "causes", "causing", "since", "due", "led", "leads", "leading", "therefore", "thus", "result", "results", "resulted", "reason", "makes", "made"]
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if t not in _STOP and len(t) > 1}


def _has_negation(text: str) -> bool:
    low = text.lower()
    return "n't" in low or any(t in _NEG for t in _WORD_RE.findall(low))


def _has_causal_cue(text: str) -> bool:
    low = text.lower()
    if any(p in low for p in ("due to", "led to", "because of", "results in", "so that")):
        return True
    return any(t in _CAUSAL for t in _WORD_RE.findall(low))


class Entailment(Protocol):
    version: str

    def classify(self, premise: str, hypothesis: str) -> tuple[str, float]: ...


class LexicalEntailment:
    """Conservative placeholder grounder. entail iff hypothesis content tokens are
    (almost) contained in the premise; contradict iff polarity mismatch on shared
    content; else neutral. High-precision / low-recall by design → biases to
    neutral (fail-closed). NOT production NLI — swap for a cross-encoder/LLM."""

    version = "lexical_v0"

    def classify(self, premise: str, hypothesis: str) -> tuple[str, float]:
        h = _content_tokens(hypothesis)
        if not h:
            return ("neutral", 0.0)
        p = _content_tokens(premise)
        overlap = len(h & p) / len(h)
        if overlap >= TAU_LOW and (_has_negation(premise) != _has_negation(hypothesis)):
            return ("contradict", round(overlap, 3))
        if overlap >= 0.9:
            return ("entail", round(overlap, 3))
        return ("neutral", round(overlap, 3))


def _calibrate(final_class: str, score: float, *, causal_no_cue: bool = False) -> Confidence:
    if final_class == "directly_supported":
        cal = min(0.95, 0.6 + 0.4 * score)
    elif causal_no_cue:
        cal = 0.35
    else:  # weakly_inferred
        cal = 0.4 + 0.2 * score
    return Confidence.from_score(raw=score, calibrated=cal)


def _verify_node(node: Node, spans_by_id: dict[str, TextSpan], ent: Entailment) -> VerifierDecision:
    reasons: list[str] = []
    # (A) schema
    if node.node_type not in NODE_TYPES:
        return VerifierDecision(node.node_id, "node", "reject", None, ["invalid_node_type"])
    if node.claim_class not in CLAIM_CLASSES:
        node.claim_class = "weakly_inferred"
        reasons.append("coerced_claim_class")

    # (B) provenance
    cited = [spans_by_id[s] for s in node.evidence if s in spans_by_id]
    has_span = len(cited) > 0
    inferred_ok = node.claim_class == "weakly_inferred" and node.inference_type in INFERENCE_TYPES
    if not has_span and not inferred_ok:
        return VerifierDecision(node.node_id, "node", "reject", None, [*reasons, "no_provenance"])

    # (C) entailment vs cited spans
    premise = " ".join(sp.text for sp in cited)
    label, score = ent.classify(premise, node.label) if cited else ("neutral", 0.0)
    comp = {"provenance": has_span, "nli": label, "nli_score": score}

    if label == "contradict":
        return VerifierDecision(node.node_id, "node", "reject", None, [*reasons, "nli_contradict"], comp, escalated=True)

    if has_span and label == "entail":
        return VerifierDecision(node.node_id, "node", "accept", "directly_supported", reasons, comp,
                                _calibrate("directly_supported", score))
    if score >= TAU_LOW and (has_span or inferred_ok):
        return VerifierDecision(node.node_id, "node", "downgrade", "weakly_inferred",
                                [*reasons, "insufficient_entailment"], comp, _calibrate("weakly_inferred", score))
    # too weak to surface (unsupported-but-plausible) -> omit, may revisit with more evidence
    return VerifierDecision(node.node_id, "node", "abstain", None, [*reasons, "insufficient_evidence"], comp)


def _verify_edge(edge: Edge, spans_by_id: dict[str, TextSpan], live_nodes: set[str]) -> VerifierDecision:
    # (A) schema + (D) dangling
    if edge.edge_type not in EDGE_TYPES:
        return VerifierDecision(edge.edge_id, "edge", "reject", None, ["invalid_edge_type"])
    if edge.src not in live_nodes or edge.dst not in live_nodes:
        return VerifierDecision(edge.edge_id, "edge", "reject", None, ["dangling_edge"])

    # (B) provenance
    cited = [spans_by_id[s] for s in edge.evidence if s in spans_by_id]
    has_span = len(cited) > 0
    inferred_ok = edge.claim_class == "weakly_inferred" and edge.inference_type in INFERENCE_TYPES
    if not has_span and not inferred_ok:
        return VerifierDecision(edge.edge_id, "edge", "reject", None, ["no_provenance"])

    premise = " ".join(sp.text for sp in cited)
    comp = {"provenance": has_span, "causal_cue": _has_causal_cue(premise) if edge.edge_type == "causal" else None}

    # causal edges require explicit causal language to be 'directly_supported'
    if edge.edge_type == "causal" and not _has_causal_cue(premise):
        return VerifierDecision(edge.edge_id, "edge", "downgrade", "weakly_inferred", ["no_causal_language"], comp,
                                _calibrate("weakly_inferred", edge.generator_confidence, causal_no_cue=True))
    if has_span:
        return VerifierDecision(edge.edge_id, "edge", "accept", "directly_supported", [], comp,
                                _calibrate("directly_supported", edge.generator_confidence))
    return VerifierDecision(edge.edge_id, "edge", "downgrade", "weakly_inferred", ["inferred_edge"], comp,
                            _calibrate("weakly_inferred", edge.generator_confidence))


def _flag_contradictions(nodes: list[Node], decisions: dict[str, VerifierDecision], ent: Entailment) -> None:
    """Cross-node contradiction: flag both (escalate), never silently merge."""
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            label, _ = ent.classify(nodes[i].label, nodes[j].label)
            if label == "contradict":
                for nid in (nodes[i].node_id, nodes[j].node_id):
                    d = decisions.get(nid)
                    if d:
                        d.escalated = True
                        if "possible_contradiction" not in d.reason_codes:
                            d.reason_codes.append("possible_contradiction")


def verify_graph(
    doc: RawDocument,
    spans: list[TextSpan],
    candidate: CandidateGraph,
    *,
    entailment: Entailment | None = None,
) -> MindmapArtifact:
    ent = entailment or LexicalEntailment()
    spans_by_id = {sp.span_id: sp for sp in spans}

    node_decisions = {n.node_id: _verify_node(n, spans_by_id, ent) for n in candidate.nodes}

    surviving_nodes: list[Node] = []
    for n in candidate.nodes:
        d = node_decisions[n.node_id]
        if d.decision in ("accept", "downgrade"):
            n.status = "verified" if d.decision == "accept" else "downgraded"
            n.claim_class = d.final_class or n.claim_class
            n.confidence = d.confidence
            surviving_nodes.append(n)

    _flag_contradictions(surviving_nodes, node_decisions, ent)
    live = {n.node_id for n in surviving_nodes}

    edge_decisions: dict[str, VerifierDecision] = {}
    surviving_edges: list[Edge] = []
    for e in candidate.edges:
        d = _verify_edge(e, spans_by_id, live)
        edge_decisions[e.edge_id] = d
        if d.decision in ("accept", "downgrade"):
            e.status = "verified" if d.decision == "accept" else "downgraded"
            e.claim_class = d.final_class or e.claim_class
            e.confidence = d.confidence
            surviving_edges.append(e)

    all_decisions = list(node_decisions.values()) + list(edge_decisions.values())
    suppressed = [
        {"claim_id": d.claim_id, "kind": d.claim_kind, "decision": d.decision, "reason_codes": d.reason_codes}
        for d in all_decisions
        if d.decision in ("abstain", "reject")
    ]

    used_spans = {s for n in surviving_nodes for s in n.evidence} | {s for e in surviving_edges for s in e.evidence}
    coverage = {
        "spans_total": len(spans),
        "spans_used": len(used_spans),
        "ratio": round(len(used_spans) / len(spans), 3) if spans else 0.0,
    }

    return MindmapArtifact(
        mindmap_id=f"mm_{doc.doc_id}",
        doc_id=doc.doc_id,
        user_id=doc.user_id,
        nodes=surviving_nodes,
        edges=surviving_edges,
        suppressed=suppressed,
        decisions=all_decisions,
        coverage=coverage,
        calibration={"n_claims": len(all_decisions), "ece": None},  # ECE measured on gold (future)
        abstained=len(surviving_nodes) == 0,
        pipeline_version=doc.pipeline_version,
        verifier_versions={"entailment": ent.version, "calibrator": "rule_v0", "rules": VERIFY_VERSION},
        created_at=datetime.now(UTC).isoformat(),
    )
