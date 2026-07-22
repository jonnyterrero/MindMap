"""Daily batch: journal entries -> verified mindmap -> upsert to Supabase.

Runs the text->verified-mindmap pipeline (Stages 1-3) over each journal entry
and writes one artifact per entry to ``mindmap_graphs``; the Next app only
reads them. Idempotent on (user_id, source_table, source_id, pipeline_version):
re-running refreshes a changed entry's row in place rather than accumulating.

``content_sha`` lets a run SKIP entries whose text is unchanged since the last
build -- the pipeline's LLM extractor is the expensive step, so this keeps the
daily job proportional to what actually changed. Dry-run prints without writing.
Pure ``build_graph_rows`` / ``filter_changed`` are unit-tested; the CLI wires
the reader + sink.

The verified artifact is the safety boundary: nodes/edges are extracted from
the user's OWN text and pass Stage-3 verification (fail-closed) before landing
here, so there is no second output gate -- an abstained entry writes a row with
``abstained=true`` and an empty graph rather than being silently dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ..graph.generate import LLMClient
from ..graph.pipeline import run_pipeline
from ..graph.verify import Entailment

USER_COL = "user_id"
ID_COL = "id"
DATE_COL = "entry_date"
CONTENT_COL = "content"


def content_sha(text: str) -> str:
    """Stable digest of the source text — the skip-unchanged key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _entry_date_str(value: Any) -> str | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def filter_changed(journal: pd.DataFrame, existing_shas: dict[str, str]) -> pd.DataFrame:
    """Drop journal rows whose content_sha already matches a persisted graph.

    ``existing_shas`` maps ``str(source_id) -> content_sha`` (from
    :func:`supabase_io.read_graph_shas`). An empty map builds everything.
    """
    if journal.empty:
        return journal
    keep = journal.apply(
        lambda r: existing_shas.get(str(r[ID_COL])) != content_sha(r[CONTENT_COL]),
        axis=1,
    )
    return journal[keep].reset_index(drop=True)


def build_graph_rows(
    journal: pd.DataFrame,
    *,
    extractor_client: LLMClient | None = None,
    entailment: Entailment | None = None,
) -> list[dict[str, Any]]:
    """Pure: one verified-mindmap upsert row per journal entry.

    ``extractor_client`` / ``entailment`` are injectable so tests run offline
    (the rule-skeleton generator + lexical grounder need no key/network).
    """
    if journal.empty:
        return []
    now_iso = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for r in journal.to_dict("records"):
        content = r[CONTENT_COL]
        if not isinstance(content, str) or not content.strip():
            continue
        art = run_pipeline(
            content,
            user_id=str(r[USER_COL]),
            source_type="journal",
            extractor_client=extractor_client,
            entailment=entailment,
        )
        rows.append(
            {
                "user_id": str(r[USER_COL]),
                "doc_id": art.doc_id,
                "mindmap_id": art.mindmap_id,
                "source_type": "journal",
                "source_table": "mindmap_journal_entries",
                "source_id": str(r[ID_COL]),
                "entry_date": _entry_date_str(r.get(DATE_COL)),
                "content_sha": content_sha(content),
                "abstained": bool(art.abstained),
                "payload": art.to_dict(),
                "pipeline_version": art.pipeline_version,
                "verifier_versions": art.verifier_versions,
                "updated_at": now_iso,
            }
        )
    return rows


def run_graph_batch(
    journal: pd.DataFrame,
    *,
    existing_shas: dict[str, str] | None = None,
    extractor_client: LLMClient | None = None,
    entailment: Entailment | None = None,
    dry_run: bool = True,
    sink: Any | None = None,
) -> list[dict[str, Any]]:
    """Filter unchanged, build rows, and (unless dry-run) upsert via ``sink``."""
    if existing_shas:
        journal = filter_changed(journal, existing_shas)
    rows = build_graph_rows(journal, extractor_client=extractor_client, entailment=entailment)
    if dry_run or sink is None:
        return rows
    sink.upsert(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="MindMap verified-mindmap batch (journal -> graph).")
    ap.add_argument("--dry-run", action="store_true", help="print rows; do not write")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic journal text instead of Supabase")
    ap.add_argument("--all", action="store_true", help="rebuild every entry (ignore content_sha skip)")
    ap.add_argument("--llm", action="store_true", help="use the Anthropic extractor + grounder (needs key)")
    args = ap.parse_args()

    extractor_client: LLMClient | None = None
    entailment: Entailment | None = None
    if args.llm:
        from ..graph.generate import AnthropicExtractor
        from ..graph.verify import make_entailment

        extractor_client = AnthropicExtractor()
        entailment = make_entailment(prefer_llm=True)

    if args.synthetic:
        journal = pd.DataFrame(
            [
                {"id": "00000000-0000-0000-0000-000000000001", "user_id": "synthetic-user",
                 "entry_date": "2026-01-01",
                 "content": "I was anxious about work and I slept badly. My migraine came back because I only slept three hours."},
                {"id": "00000000-0000-0000-0000-000000000002", "user_id": "synthetic-user",
                 "entry_date": "2026-01-02",
                 "content": "Felt calmer today after a walk outside. Work is still a black hole swallowing my time."},
            ]
        )
        sink = None
        existing: dict[str, str] = {}
    else:
        from .supabase_io import (
            SupabaseGraphSink,
            get_client,
            read_graph_shas,
            read_journal_entries,
        )

        client = get_client()
        journal = read_journal_entries(client)
        existing = {} if args.all else read_graph_shas(client)
        sink = None if args.dry_run else SupabaseGraphSink(client)

    rows = run_graph_batch(
        journal,
        existing_shas=existing,
        extractor_client=extractor_client,
        entailment=entailment,
        dry_run=args.dry_run,
        sink=sink,
    )
    n_abstained = sum(1 for r in rows if r["abstained"])
    n_users = journal[USER_COL].nunique() if not journal.empty else 0
    print(f"built {len(rows)} mindmaps ({n_abstained} abstained) for {n_users} users")
    if args.dry_run:
        for r in rows[:3]:
            print(json.dumps(r)[:600])
        print("(dry-run: nothing written)")


if __name__ == "__main__":
    main()
