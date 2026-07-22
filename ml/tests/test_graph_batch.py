"""Verified-mindmap batch writer: row shape, idempotency key, skip-unchanged."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mindmap_ml.serving.graph_batch import (
    build_graph_rows,
    content_sha,
    filter_changed,
    run_graph_batch,
)
from mindmap_ml.serving.supabase_io import CollectingSink


@pytest.fixture
def journal() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "user_id": "22222222-2222-2222-2222-222222222222",
                "entry_date": "2026-01-01",
                "content": "I was anxious about work and I slept badly.",
            },
            {
                "id": "33333333-3333-3333-3333-333333333333",
                "user_id": "22222222-2222-2222-2222-222222222222",
                "entry_date": "2026-01-02",
                "content": "Work is a black hole swallowing all my time.",
            },
        ]
    )


# --------------------------- row shape -------------------------------------- #
def test_one_row_per_entry_with_required_columns(journal) -> None:
    rows = build_graph_rows(journal)
    assert len(rows) == 2
    required = {
        "user_id", "doc_id", "mindmap_id", "source_type", "source_table",
        "source_id", "entry_date", "content_sha", "abstained", "payload",
        "pipeline_version", "verifier_versions", "updated_at",
    }
    for r in rows:
        assert required <= set(r)
        assert r["source_table"] == "mindmap_journal_entries"
        assert r["source_type"] == "journal"
        assert r["mindmap_id"] == f"mm_{r['doc_id']}"


def test_source_id_and_date_carried_from_journal(journal) -> None:
    rows = build_graph_rows(journal)
    by_source = {r["source_id"]: r for r in rows}
    assert "11111111-1111-1111-1111-111111111111" in by_source
    assert by_source["11111111-1111-1111-1111-111111111111"]["entry_date"] == "2026-01-01"


def test_rows_are_json_serializable(journal) -> None:
    for r in build_graph_rows(journal):
        json.dumps(r)  # payload/verifier_versions must be plain jsonable


def test_payload_carries_verified_graph(journal) -> None:
    rows = build_graph_rows(journal)
    for r in rows:
        payload = r["payload"]
        assert {"nodes", "edges", "coverage"} <= set(payload)
        # abstained rows carry an empty node list; non-abstained carry nodes
        assert r["abstained"] == (len(payload["nodes"]) == 0)


# --------------------------- idempotency / skip ----------------------------- #
def test_content_sha_stable_and_short() -> None:
    s = content_sha("hello world")
    assert s == content_sha("hello world") and len(s) == 16
    assert s != content_sha("hello world!")


def test_filter_changed_skips_matching_sha(journal) -> None:
    existing = {row["id"]: content_sha(row["content"]) for _, row in journal.iterrows()}
    assert filter_changed(journal, existing).empty  # nothing changed -> nothing to build


def test_filter_changed_keeps_edited_and_new(journal) -> None:
    existing = {
        "11111111-1111-1111-1111-111111111111": "staleoldsha000000",  # edited
        # second entry absent -> new
    }
    changed = filter_changed(journal, existing)
    assert set(changed["id"]) == {
        "11111111-1111-1111-1111-111111111111",
        "33333333-3333-3333-3333-333333333333",
    }


# --------------------------- run wiring ------------------------------------- #
def test_dry_run_writes_nothing(journal) -> None:
    sink = CollectingSink()
    rows = run_graph_batch(journal, dry_run=True, sink=sink)
    assert rows and sink.rows == []


def test_real_run_upserts(journal) -> None:
    sink = CollectingSink()
    rows = run_graph_batch(journal, dry_run=False, sink=sink)
    assert len(sink.rows) == len(rows) == 2


def test_run_respects_existing_shas(journal) -> None:
    existing = {row["id"]: content_sha(row["content"]) for _, row in journal.iterrows()}
    sink = CollectingSink()
    rows = run_graph_batch(journal, existing_shas=existing, dry_run=False, sink=sink)
    assert rows == [] and sink.rows == []  # all unchanged -> no work


def test_empty_journal() -> None:
    assert build_graph_rows(pd.DataFrame()) == []
    assert run_graph_batch(pd.DataFrame(), dry_run=False, sink=CollectingSink()) == []


def test_blank_content_skipped() -> None:
    df = pd.DataFrame([{"id": "a", "user_id": "u", "entry_date": "2026-01-01", "content": "   "}])
    assert build_graph_rows(df) == []
