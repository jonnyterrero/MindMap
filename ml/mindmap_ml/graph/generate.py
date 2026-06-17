"""Stage 2 — Structured Generation (probabilistic/hybrid).

Proposes a CANDIDATE concept graph from Stage-1 spans. Candidates only — the
generator cannot self-approve; Stage 3 verifies independently. The foundation LLM
(claude-opus-4-8) emits strict JSON, each node/edge citing span_ids. A
deterministic rule-skeleton is the fallback when the LLM is unavailable or returns
unparseable output (fail-safe, fully testable). Embedding dedup merges near-duplicate
concepts.

The LLM client is injected so this is testable without a key/network.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from ..evidence.index import TfidfEmbedder
from .schema import (
    CandidateGraph,
    Edge,
    Node,
    RawDocument,
    TextSpan,
)

EXTRACTOR_MODEL = "claude-opus-4-8"  # matches lib/ai-reflection.ts
GENERATOR_VERSION = f"gen_v0:{EXTRACTOR_MODEL}"
_DEDUP_THRESHOLD = 0.85
_LABEL_MAXLEN = 80

SYSTEM_PROMPT = """You extract a concept graph from a person's own writing (journal/brainstorm/notes).

Output STRICT JSON only, matching:
{"nodes":[{"id":"n1","label":str,"type":<node_type>,"evidence":[span_id,...],
           "claim_class":"directly_supported"|"weakly_inferred","inference_type":<type|null>,"confidence":0..1}],
 "edges":[{"src":"n1","dst":"n2","type":<edge_type>,"evidence":[span_id,...],
           "claim_class":...,"inference_type":...,"confidence":0..1}]}

node_type: theme|entity|goal|emotion|event|value|question
edge_type: causal|temporal|thematic|contrast|elaboration|part_of
inference_type: explicit|co_occurrence|semantic|prior_based|null

Rules:
- Every node and edge MUST cite span_id(s) it is grounded in, from the spans provided.
- If something is inferred (not literally stated), set claim_class="weakly_inferred" + an
  inference_type, and still cite the spans that motivated it.
- A causal edge requires explicit causal language in the cited span; otherwise use
  thematic/temporal/co_occurrence.
- Do NOT invent entities, themes, or causal links the text does not support.
- Prefer fewer, well-grounded nodes over many speculative ones.
- You are a CANDIDATE generator; a separate verifier checks you. Do not assert confidence
  you cannot justify from the spans. Output JSON only, no prose."""


class LLMClient(Protocol):
    def generate_json(self, system: str, user: str) -> str: ...


class AnthropicExtractor:
    """Lazy Anthropic client; returns the model's JSON text."""

    def __init__(self, model: str = EXTRACTOR_MODEL) -> None:
        self.model = model

    def generate_json(self, system: str, user: str) -> str:  # pragma: no cover - network
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set; extractor unavailable.")
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


def build_user_prompt(spans: list[TextSpan]) -> str:
    lines = ["SPANS (cite these ids):"]
    lines += [f"  {sp.span_id}: {sp.text}" for sp in spans]
    lines.append("\nReturn the JSON graph now.")
    return "\n".join(lines)


def _coerce_str(v: Any, default: str = "") -> str:
    return v if isinstance(v, str) else default


def _coerce_conf(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


def parse_candidate_graph(raw_json: str, doc_id: str, valid_span_ids: set[str]) -> CandidateGraph:
    """Parse the LLM JSON into typed candidates. Lenient: coerces, namespaces ids,
    drops invalid span citations and dangling edges (recorded as missing). Raises on
    unparseable JSON (caller falls back to the rule skeleton)."""
    data = json.loads(raw_json)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")

    nodes: list[Node] = []
    id_map: dict[str, str] = {}  # llm id -> our node_id
    for i, raw in enumerate(data.get("nodes", []) or []):
        if not isinstance(raw, dict):
            continue
        nid = f"nd_{doc_id}_{i}"
        llm_id = _coerce_str(raw.get("id"), nid)
        id_map[llm_id] = nid
        ev = [s for s in (raw.get("evidence") or []) if s in valid_span_ids]
        nodes.append(
            Node(
                node_id=nid,
                label=_coerce_str(raw.get("label"))[:_LABEL_MAXLEN].strip() or "(unlabeled)",
                node_type=_coerce_str(raw.get("type"), "theme"),
                evidence=ev,
                generator_confidence=_coerce_conf(raw.get("confidence")),
                claim_class=_coerce_str(raw.get("claim_class"), "directly_supported"),
                inference_type=raw.get("inference_type") if isinstance(raw.get("inference_type"), str) else None,
            )
        )

    edges: list[Edge] = []
    for i, raw in enumerate(data.get("edges", []) or []):
        if not isinstance(raw, dict):
            continue
        src = id_map.get(_coerce_str(raw.get("src")))
        dst = id_map.get(_coerce_str(raw.get("dst")))
        if not src or not dst or src == dst:
            continue  # drop dangling / self edges
        ev = [s for s in (raw.get("evidence") or []) if s in valid_span_ids]
        edges.append(
            Edge(
                edge_id=f"ed_{doc_id}_{i}",
                src=src,
                dst=dst,
                edge_type=_coerce_str(raw.get("type"), "thematic"),
                evidence=ev,
                generator_confidence=_coerce_conf(raw.get("confidence")),
                claim_class=_coerce_str(raw.get("claim_class"), "directly_supported"),
                inference_type=raw.get("inference_type") if isinstance(raw.get("inference_type"), str) else None,
            )
        )

    return CandidateGraph(doc_id=doc_id, nodes=nodes, edges=edges, generator_version=GENERATOR_VERSION, method="llm")


def rule_skeleton(doc: RawDocument, spans: list[TextSpan]) -> CandidateGraph:
    """Deterministic fallback: one grounded node per span, no edges. Degraded but
    every node maps to exactly one real span (always passes provenance)."""
    nodes = [
        Node(
            node_id=f"nd_{doc.doc_id}_{i}",
            label=(sp.text[:_LABEL_MAXLEN].strip() or "(unlabeled)"),
            node_type="question" if sp.text.rstrip().endswith("?") else "theme",
            evidence=[sp.span_id],
            generator_confidence=0.5,
            claim_class="directly_supported",
        )
        for i, sp in enumerate(spans)
    ]
    return CandidateGraph(doc_id=doc.doc_id, nodes=nodes, edges=[], generator_version="rule_skeleton_v0", method="rule_skeleton")


def _dedup_nodes(nodes: list[Node]) -> tuple[list[Node], dict[str, str]]:
    remap = {n.node_id: n.node_id for n in nodes}
    if len(nodes) < 2:
        return nodes, remap
    emb = TfidfEmbedder()
    mat = emb.fit_transform([n.label for n in nodes])
    sims = mat @ mat.T
    used = [False] * len(nodes)
    kept: list[Node] = []
    for i, n in enumerate(nodes):
        if used[i]:
            continue
        used[i] = True
        kept.append(n)
        for j in range(i + 1, len(nodes)):
            if used[j] or nodes[j].node_type != n.node_type:
                continue
            if float(sims[i, j]) >= _DEDUP_THRESHOLD:
                used[j] = True
                remap[nodes[j].node_id] = n.node_id
                for sp in nodes[j].evidence:
                    if sp not in n.evidence:
                        n.evidence.append(sp)
                if nodes[j].label != n.label and nodes[j].label not in n.aliases:
                    n.aliases.append(nodes[j].label)
    return kept, remap


def _remap_edges(edges: list[Edge], remap: dict[str, str]) -> list[Edge]:
    out: list[Edge] = []
    seen: set[tuple[str, str, str]] = set()
    for e in edges:
        e.src = remap.get(e.src, e.src)
        e.dst = remap.get(e.dst, e.dst)
        if e.src == e.dst:
            continue
        key = (e.src, e.dst, e.edge_type)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def generate_candidates(
    doc: RawDocument,
    spans: list[TextSpan],
    *,
    client: LLMClient | None = None,
    dedup: bool = True,
) -> CandidateGraph:
    """Stage-2 entry point. With no client (or on parse failure) uses the rule
    skeleton — the pipeline degrades, never crashes."""
    if not spans:
        return CandidateGraph(doc_id=doc.doc_id, generator_version=GENERATOR_VERSION, method="empty")

    if client is None:
        graph = rule_skeleton(doc, spans)
    else:
        valid_ids = {sp.span_id for sp in spans}
        try:
            graph = parse_candidate_graph(client.generate_json(SYSTEM_PROMPT, build_user_prompt(spans)), doc.doc_id, valid_ids)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            graph = rule_skeleton(doc, spans)  # fail-safe

    if dedup:
        graph.nodes, remap = _dedup_nodes(graph.nodes)
        graph.edges = _remap_edges(graph.edges, remap)
    return graph
