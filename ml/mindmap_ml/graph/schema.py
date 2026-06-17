"""Data schema for the verified-mindmap pipeline (all stages).

Stage 1: RawDocument, TextSpan (offset-addressable evidence).
Stage 2: Node, Edge, CandidateGraph (generator proposals — CANDIDATES only).
Stage 3: Confidence, VerifierDecision, MindmapArtifact (verified output).

Offset convention (the core guarantee, enforced in ingest):
  * ``RawDocument.raw_text`` is **NFC-canonicalized**; ``raw_start``/``raw_end``
    index into it. Highlight the original via ``raw_text[raw_start:raw_end]``.
  * ``norm_text[norm_start:norm_end] == TextSpan.text`` exactly; re-normalizing
    ``raw_text[raw_start:raw_end]`` reproduces it (reversible). Spans carry both
    offset pairs, so no separate char map is persisted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

GRAPH_PIPELINE_VERSION = "graph_pipeline_v0"

SOURCE_TYPES: tuple[str, ...] = ("journal", "brainstorm", "notes", "voice_transcript")
UNIT_TYPES: tuple[str, ...] = ("sentence", "clause", "fragment")

NODE_TYPES: tuple[str, ...] = ("theme", "entity", "goal", "emotion", "event", "value", "question")
EDGE_TYPES: tuple[str, ...] = ("causal", "temporal", "thematic", "contrast", "elaboration", "part_of")
INFERENCE_TYPES: tuple[str, ...] = ("explicit", "co_occurrence", "semantic", "prior_based")
CLAIM_CLASSES: tuple[str, ...] = ("directly_supported", "weakly_inferred", "unverifiable")
STATUSES: tuple[str, ...] = ("candidate", "verified", "downgraded", "suppressed", "abstained")


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TextSpan:
    span_id: str
    doc_id: str
    text: str
    norm_start: int
    norm_end: int
    raw_start: int
    raw_end: int
    unit_type: str
    order: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RawDocument:
    doc_id: str
    user_id: str
    source_type: str
    raw_text: str  # NFC-canonical; raw offsets index into this
    norm_text: str  # normalized view; norm offsets index into this
    language: str
    language_confidence: float
    created_at: str
    pipeline_version: str = GRAPH_PIPELINE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Stage 3 confidence (attached to verified claims)
# --------------------------------------------------------------------------- #
@dataclass
class Confidence:
    raw_score: float  # pre-calibration signal
    calibrated: float  # 0..1
    bucket: str  # low | medium | high
    calibrator_version: str = "rule_v0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_score(cls, raw: float, calibrated: float, version: str = "rule_v0") -> Confidence:
        c = max(0.0, min(1.0, calibrated))
        bucket = "high" if c >= 0.7 else "medium" if c >= 0.45 else "low"
        return cls(raw_score=round(raw, 3), calibrated=round(c, 3), bucket=bucket, calibrator_version=version)


# --------------------------------------------------------------------------- #
# Stage 2 candidates / verified graph elements
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    node_id: str
    label: str
    node_type: str
    evidence: list[str] = field(default_factory=list)  # span_ids
    aliases: list[str] = field(default_factory=list)
    generator_confidence: float = 0.5
    claim_class: str = "directly_supported"
    inference_type: str | None = None
    status: str = "candidate"
    confidence: Confidence | None = None  # set by verifier

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["confidence"] = self.confidence.to_dict() if self.confidence else None
        return d


@dataclass
class Edge:
    edge_id: str
    src: str  # node_id
    dst: str  # node_id
    edge_type: str
    evidence: list[str] = field(default_factory=list)  # span_ids
    generator_confidence: float = 0.5
    claim_class: str = "directly_supported"
    inference_type: str | None = None
    status: str = "candidate"
    confidence: Confidence | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["confidence"] = self.confidence.to_dict() if self.confidence else None
        return d


@dataclass
class CandidateGraph:
    doc_id: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    generator_version: str = "unknown"
    method: str = "llm"  # llm | rule_skeleton

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "generator_version": self.generator_version,
            "method": self.method,
        }


# --------------------------------------------------------------------------- #
# Stage 3 verifier decisions + final artifact
# --------------------------------------------------------------------------- #
@dataclass
class VerifierDecision:
    claim_id: str  # node_id | edge_id
    claim_kind: str  # node | edge
    decision: str  # accept | downgrade | abstain | reject
    final_class: str | None
    reason_codes: list[str] = field(default_factory=list)
    component_scores: dict[str, Any] = field(default_factory=dict)
    confidence: Confidence | None = None
    escalated: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["confidence"] = self.confidence.to_dict() if self.confidence else None
        return d


@dataclass
class MindmapArtifact:
    mindmap_id: str
    doc_id: str
    user_id: str
    nodes: list[Node] = field(default_factory=list)  # verified / downgraded only
    edges: list[Edge] = field(default_factory=list)
    suppressed: list[dict[str, Any]] = field(default_factory=list)  # {claim_id, kind, reason_codes}
    decisions: list[VerifierDecision] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    abstained: bool = False
    pipeline_version: str = GRAPH_PIPELINE_VERSION
    verifier_versions: dict[str, str] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mindmap_id": self.mindmap_id,
            "doc_id": self.doc_id,
            "user_id": self.user_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "suppressed": self.suppressed,
            "decisions": [d.to_dict() for d in self.decisions],
            "coverage": self.coverage,
            "calibration": self.calibration,
            "abstained": self.abstained,
            "pipeline_version": self.pipeline_version,
            "verifier_versions": self.verifier_versions,
            "created_at": self.created_at,
        }
