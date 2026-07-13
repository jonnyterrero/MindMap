"""Batch: run the verified-mindmap pipeline per source text → upsert to Supabase.

Two sources: plaintext journal entries (``mindmap_journal_entries``) and an
Obsidian vault (``--vault``). The app only reads ``mindmap_graphs``; this is
the writer. Idempotent on (user_id, doc_id, pipeline_version) — doc_id is a
content hash, so re-running over unchanged text rewrites the same row. Pure
``build_graph_rows`` / ``build_vault_rows`` are unit-tested; ``main`` wires
the sink. With no --llm flag (or no key) the pipeline degrades to the
deterministic rule skeleton + lexical grounder and stays fail-closed.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from ..graph.generate import LLMClient
from ..graph.ingest import digest
from ..graph.obsidian import ObsidianNote, load_vault
from ..graph.pipeline import run_pipeline
from ..graph.schema import MindmapArtifact
from ..graph.verify import Entailment


def _evidence_texts(text: str, *, user_id: str, source_type: str, artifact: MindmapArtifact) -> dict[str, str]:
    """span_id -> quoted text for every span a surviving claim cites, so the
    app can show provenance without re-running Stage 1. digest is deterministic,
    so span ids here match the ones inside the artifact."""
    _, spans = digest(text, user_id=user_id, source_type=source_type)
    cited: set[str] = set()
    for n in artifact.nodes:
        cited.update(n.evidence)
    for e in artifact.edges:
        cited.update(e.evidence)
    return {sp.span_id: sp.text for sp in spans if sp.span_id in cited}


def _artifact_row(artifact: MindmapArtifact, *, source_type: str, now_iso: str) -> dict[str, Any]:
    return {
        "user_id": artifact.user_id,
        "doc_id": artifact.doc_id,
        "mindmap_id": artifact.mindmap_id,
        "source_type": source_type,
        "abstained": bool(artifact.abstained),
        "payload": artifact.to_dict(),
        "pipeline_version": artifact.pipeline_version,
        "updated_at": now_iso,
    }


def build_graph_rows(
    entries: list[dict[str, Any]],
    *,
    extractor_client: LLMClient | None = None,
    entailment: Entailment | None = None,
) -> list[dict[str, Any]]:
    """One verified-graph row per journal entry with non-empty plaintext."""
    now_iso = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for e in entries:
        text = (e.get("content") or "").strip()
        if not text:
            continue
        artifact = run_pipeline(
            text,
            user_id=str(e["user_id"]),
            source_type="journal",
            extractor_client=extractor_client,
            entailment=entailment,
        )
        row = _artifact_row(artifact, source_type="journal", now_iso=now_iso)
        row["payload"]["evidence_texts"] = _evidence_texts(
            text, user_id=str(e["user_id"]), source_type="journal", artifact=artifact
        )
        row["payload"]["source_meta"] = {
            "journal_entry_id": str(e.get("id", "")),
            "entry_date": str(e.get("entry_date", "")),
            "title": e.get("title") or "",
        }
        rows.append(row)
    return rows


def build_vault_rows(
    notes: list[ObsidianNote],
    *,
    user_id: str,
    extractor_client: LLMClient | None = None,
    entailment: Entailment | None = None,
) -> list[dict[str, Any]]:
    """One verified-graph row per Obsidian note with non-empty prose."""
    now_iso = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for note in notes:
        if not note.text:
            continue
        artifact = run_pipeline(
            note.text,
            user_id=user_id,
            source_type="notes",
            extractor_client=extractor_client,
            entailment=entailment,
        )
        row = _artifact_row(artifact, source_type="notes", now_iso=now_iso)
        row["payload"]["evidence_texts"] = _evidence_texts(
            note.text, user_id=user_id, source_type="notes", artifact=artifact
        )
        row["payload"]["source_meta"] = {
            "obsidian_path": note.path,
            "title": note.title,
            "tags": note.tags,
            "wikilinks": note.wikilinks,
        }
        rows.append(row)
    return rows


def run_graph_batch(
    entries: list[dict[str, Any]], *, dry_run: bool = True, sink: Any | None = None
) -> list[dict[str, Any]]:
    rows = build_graph_rows(entries)
    if dry_run or sink is None:
        return rows
    sink.upsert(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="MindMap verified-graph batch.")
    ap.add_argument("--dry-run", action="store_true", help="print rows; do not write")
    ap.add_argument("--vault", type=str, default=None, help="path to an Obsidian vault (notes mode)")
    ap.add_argument("--user-id", type=str, default=None, help="owner user_id (required with --vault)")
    ap.add_argument("--llm", action="store_true", help="use the Anthropic extractor (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args()

    extractor: LLMClient | None = None
    if args.llm:
        from ..graph.generate import AnthropicExtractor

        extractor = AnthropicExtractor()

    if args.vault:
        if not args.user_id:
            ap.error("--vault requires --user-id")
        rows = build_vault_rows(load_vault(args.vault), user_id=args.user_id, extractor_client=extractor)
        sink = None
        if not args.dry_run:
            from .supabase_io import SupabaseGraphsSink, get_client

            sink = SupabaseGraphsSink(get_client())
    else:
        from .supabase_io import SupabaseGraphsSink, get_client, read_journal_entries

        client = get_client()
        entries = read_journal_entries(client)
        rows = build_graph_rows(entries, extractor_client=extractor)
        sink = None if args.dry_run else SupabaseGraphsSink(client)

    if sink is not None:
        sink.upsert(rows)
    n_abstained = sum(1 for r in rows if r["abstained"])
    print(f"built {len(rows)} graphs ({n_abstained} abstained)")
    if args.dry_run:
        for r in rows[:3]:
            print(json.dumps(r)[:600])
        print("(dry-run: nothing written)")


if __name__ == "__main__":
    main()
