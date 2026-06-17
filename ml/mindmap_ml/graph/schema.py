"""Stage-1 data schema for the verified-mindmap pipeline.

Only the ingest/digest objects live here (RawDocument, TextSpan). Stage-2/3
objects (Concept, Node, Edge, VerifierDecision, MindmapArtifact, ...) are added
in their own modules as those stages land. See ml/MINDMAP_PIPELINE_DESIGN.md §6.

Offset convention (the core guarantee):
  * ``RawDocument.raw_text`` is **NFC-canonicalized**; all ``raw_start``/``raw_end``
    index into it. Highlighting the original is ``raw_text[raw_start:raw_end]``.
  * ``RawDocument.norm_text`` is the normalized view; ``norm_start``/``norm_end``
    index into it and ``norm_text[norm_start:norm_end] == TextSpan.text`` exactly.
  * Reversibility: re-normalizing ``raw_text[raw_start:raw_end]`` reproduces
    ``TextSpan.text`` (tested). Spans carry BOTH offset pairs, so no separate
    char map needs to be persisted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

GRAPH_PIPELINE_VERSION = "graph_pipeline_v0"

SOURCE_TYPES: tuple[str, ...] = ("journal", "brainstorm", "notes", "voice_transcript")
UNIT_TYPES: tuple[str, ...] = ("sentence", "clause", "fragment")


@dataclass(frozen=True)
class TextSpan:
    span_id: str
    doc_id: str
    text: str
    norm_start: int
    norm_end: int
    raw_start: int
    raw_end: int
    unit_type: str  # sentence | clause | fragment
    order: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RawDocument:
    doc_id: str
    user_id: str
    source_type: str  # one of SOURCE_TYPES
    raw_text: str  # NFC-canonical; raw offsets index into this
    norm_text: str  # normalized view; norm offsets index into this
    language: str  # "en" | "unknown" (Stage-1 heuristic)
    language_confidence: float
    created_at: str  # ISO-8601
    pipeline_version: str = GRAPH_PIPELINE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
