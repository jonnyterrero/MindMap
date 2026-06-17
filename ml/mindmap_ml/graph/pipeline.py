"""End-to-end orchestrator: raw text -> verified MindmapArtifact.

Stage 1 (digest) -> Stage 2 (generate candidates) -> Stage 3 (external verify).
Both the extractor client and the entailment grounder are injectable, so the whole
pipeline runs offline/deterministically in tests (no key, no network).
"""

from __future__ import annotations

from .generate import LLMClient, generate_candidates
from .ingest import digest
from .schema import MindmapArtifact
from .verify import Entailment, verify_graph


def run_pipeline(
    raw_text: str,
    *,
    user_id: str,
    source_type: str = "journal",
    extractor_client: LLMClient | None = None,
    entailment: Entailment | None = None,
    dedup: bool = True,
) -> MindmapArtifact:
    doc, spans = digest(raw_text, user_id=user_id, source_type=source_type)
    candidate = generate_candidates(doc, spans, client=extractor_client, dedup=dedup)
    return verify_graph(doc, spans, candidate, entailment=entailment)
