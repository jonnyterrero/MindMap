"""Verifier evaluation harness.

Runs the Stage-3 verifier against the hand-authored gold set and reports the
confusion that matters for a fail-closed system:

  true_accept  : supported claim surfaced (accept/downgrade)   — good
  false_accept : UNSUPPORTED claim surfaced                    — the dangerous error
  false_reject : supported claim blocked (abstain/reject)      — recall cost
  true_reject  : unsupported claim blocked                     — good

Headline = **false_accept_rate** (fraction of unsupported claims that slipped
through). Also precision/recall/F1, per-category breakdown, and Brier/ECE on the
calibrated confidence of surfaced claims (reusing eval/metrics).

    uv run python -m mindmap_ml.graph.evaluate
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from ..eval.metrics import brier_score, expected_calibration_error
from .gold import GOLD_CASES, GoldCase
from .ingest import digest
from .schema import CandidateGraph, Edge, Node
from .verify import Entailment, LexicalEntailment, make_entailment, verify_graph


def _find_span_id(spans: list, contains: str | None) -> list[str]:
    if not contains:
        return []
    needle = contains.lower()
    for sp in spans:
        if needle in sp.text.lower():
            return [sp.span_id]
    return []


def _candidate_from_gold(doc, spans, case: GoldCase) -> CandidateGraph:
    nodes = [
        Node(
            node_id=f"nd_{doc.doc_id}_{i}",
            label=c.label,
            node_type=c.node_type,
            evidence=_find_span_id(spans, c.evidence_contains),
            generator_confidence=0.9,  # generator is confident; verifier must not defer to that
            claim_class="directly_supported",
        )
        for i, c in enumerate(case.claims)
    ]
    edges = [
        Edge(
            edge_id=f"ed_{doc.doc_id}_{j}",
            src=f"nd_{doc.doc_id}_{e.src}",
            dst=f"nd_{doc.doc_id}_{e.dst}",
            edge_type=e.edge_type,
            evidence=[sid for loc in e.evidence_contains for sid in _find_span_id(spans, loc)],
            generator_confidence=0.9,
            claim_class="directly_supported",
        )
        for j, e in enumerate(case.edges)
    ]
    return CandidateGraph(doc_id=doc.doc_id, nodes=nodes, edges=edges)


@dataclass
class VerifierEvalReport:
    entailment_version: str
    n_claims: int
    true_accept: int
    false_accept: int
    false_reject: int
    true_reject: int
    precision: float
    recall: float
    f1: float
    false_accept_rate: float
    block_rate: float
    brier: float
    ece: float
    per_category: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def evaluate(
    cases: tuple[GoldCase, ...] = GOLD_CASES, *, entailment: Entailment | None = None
) -> VerifierEvalReport:
    ent = entailment or LexicalEntailment()
    ta = fa = fr = tr = 0
    per: dict[str, dict[str, int]] = defaultdict(lambda: {"ta": 0, "fa": 0, "fr": 0, "tr": 0})
    conf_scores: list[float] = []
    conf_labels: list[float] = []

    for case in cases:
        doc, spans = digest(case.text, user_id="gold")
        art = verify_graph(doc, spans, _candidate_from_gold(doc, spans, case), entailment=ent)
        surfaced = {n.node_id: n for n in art.nodes}
        surfaced_edges = {e.edge_id: e for e in art.edges}

        # (claim_id, gold verdict, category, surfaced element or None)
        scored: list[tuple[bool, str, object]] = []
        for i, claim in enumerate(case.claims):
            nid = f"nd_{doc.doc_id}_{i}"
            scored.append((claim.supported, claim.category, surfaced.get(nid)))
        for j, gold_edge in enumerate(case.edges):
            eid = f"ed_{doc.doc_id}_{j}"
            scored.append((gold_edge.supported, gold_edge.category, surfaced_edges.get(eid)))

        for supported, category, element in scored:
            is_surfaced = element is not None
            cat = per[category]
            if supported and is_surfaced:
                ta += 1
                cat["ta"] += 1
            elif (not supported) and is_surfaced:
                fa += 1
                cat["fa"] += 1
            elif supported and not is_surfaced:
                fr += 1
                cat["fr"] += 1
            else:
                tr += 1
                cat["tr"] += 1
            if element is not None and element.confidence is not None:
                conf_scores.append(element.confidence.calibrated)
                conf_labels.append(1.0 if supported else 0.0)

    precision = _safe_div(ta, ta + fa)
    recall = _safe_div(ta, ta + fr)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision == precision and recall == recall else float("nan")
    return VerifierEvalReport(
        entailment_version=ent.version,
        n_claims=ta + fa + fr + tr,
        true_accept=ta, false_accept=fa, false_reject=fr, true_reject=tr,
        precision=round(precision, 3) if precision == precision else precision,
        recall=round(recall, 3) if recall == recall else recall,
        f1=round(f1, 3) if f1 == f1 else f1,
        false_accept_rate=round(_safe_div(fa, fa + tr), 3),
        block_rate=round(_safe_div(fr + tr, ta + fa + fr + tr), 3),
        brier=round(brier_score(conf_labels, conf_scores), 3) if conf_scores else float("nan"),
        ece=round(expected_calibration_error(conf_labels, conf_scores, 5), 3) if conf_scores else float("nan"),
        per_category=dict(per),
    )


def format_report(r: VerifierEvalReport) -> str:
    lines = [
        f"Verifier eval — entailment={r.entailment_version}  claims={r.n_claims}",
        f"  TA={r.true_accept} FA={r.false_accept} FR={r.false_reject} TR={r.true_reject}",
        f"  precision={r.precision} recall={r.recall} f1={r.f1}",
        f"  FALSE-ACCEPT RATE={r.false_accept_rate}  block_rate={r.block_rate}",
        f"  brier={r.brier} ece={r.ece}",
        "  per-category (ta/fa/fr/tr):",
    ]
    for cat, c in sorted(r.per_category.items()):
        lines.append(f"    {cat:<14} {c['ta']}/{c['fa']}/{c['fr']}/{c['tr']}")
    lines.append("  (false-accept is the safety-critical error; lower is better)")
    return "\n".join(lines)


def main() -> None:
    import sys

    use_llm = "--llm" in sys.argv

    if use_llm:
        ent = make_entailment(prefer_llm=True)
        if not ent.version.startswith("llm"):
            print("ANTHROPIC_API_KEY not set — cannot run with --llm.", file=sys.stderr)
            sys.exit(1)
    else:
        ent = LexicalEntailment()

    print(format_report(evaluate(entailment=ent)))


if __name__ == "__main__":
    main()
