"""Graph batch — journal/vault text in, idempotent mindmap_graphs rows out.

Runs fully offline (rule skeleton + lexical grounder). The upsert key
(user_id, doc_id, pipeline_version) must be stable across re-runs of the same
content, and empty/encrypted sources must produce no rows, never crashes.
"""

from mindmap_ml.graph.obsidian import parse_note
from mindmap_ml.graph.verify import LexicalEntailment
from mindmap_ml.serving.graph_batch import build_graph_rows, build_vault_rows, run_graph_batch
from mindmap_ml.serving.supabase_io import GRAPHS_CONFLICT, CollectingSink

ENTRIES = [
    {
        "id": "e1",
        "user_id": "u1",
        "entry_date": "2026-07-10",
        "title": "rough day",
        "content": "I slept badly and felt anxious. Work was stressful.",
    },
    {"id": "e2", "user_id": "u1", "entry_date": "2026-07-11", "title": "", "content": "   "},
    {"id": "e3", "user_id": "u2", "entry_date": "2026-07-11", "title": None, "content": None},
    {
        "id": "e4",
        "user_id": "u2",
        "entry_date": "2026-07-12",
        "title": "better",
        "content": "Took a walk. Felt calmer after.",
    },
]


def test_rows_shape_and_skipping():
    rows = build_graph_rows(ENTRIES, entailment=LexicalEntailment())
    assert len(rows) == 2  # blank + encrypted-out entries skipped
    for row in rows:
        assert set(GRAPHS_CONFLICT.split(",")) <= set(row)
        assert row["source_type"] == "journal"
        assert row["payload"]["mindmap_id"] == row["mindmap_id"]
        assert row["payload"]["source_meta"]["journal_entry_id"] in ("e1", "e4")
    assert {r["user_id"] for r in rows} == {"u1", "u2"}


def test_idempotent_doc_id_across_runs():
    a = build_graph_rows(ENTRIES[:1], entailment=LexicalEntailment())
    b = build_graph_rows(ENTRIES[:1], entailment=LexicalEntailment())
    assert a[0]["doc_id"] == b[0]["doc_id"]
    assert a[0]["pipeline_version"] == b[0]["pipeline_version"]


def test_payload_is_fail_closed_artifact():
    rows = build_graph_rows(ENTRIES[:1], entailment=LexicalEntailment())
    payload = rows[0]["payload"]
    for node in payload["nodes"]:
        assert node["status"] in ("verified", "downgraded")
        assert node["confidence"] is not None
    assert "decisions" in payload and "suppressed" in payload


def test_evidence_texts_cover_all_cited_spans():
    rows = build_graph_rows(ENTRIES[:1], entailment=LexicalEntailment())
    payload = rows[0]["payload"]
    quotes = payload["evidence_texts"]
    cited = {sid for n in payload["nodes"] for sid in n["evidence"]}
    cited |= {sid for e in payload["edges"] for sid in e["evidence"]}
    assert cited and set(quotes) == cited
    assert all(q.strip() for q in quotes.values())


def test_vault_rows_carry_obsidian_meta():
    note = parse_note(
        "---\ntags: [sleep]\n---\nSlept four hours. Felt foggy at work. See [[Sleep Hygiene]].",
        title="2026-07-12",
    )
    note.path = "daily/2026-07-12.md"
    empty = parse_note("---\ntags: [x]\n---\n", title="empty")
    rows = build_vault_rows([note, empty], user_id="u9", entailment=LexicalEntailment())
    assert len(rows) == 1  # structure-only note skipped
    meta = rows[0]["payload"]["source_meta"]
    assert meta == {
        "obsidian_path": "daily/2026-07-12.md",
        "title": "2026-07-12",
        "tags": ["sleep"],
        "wikilinks": ["Sleep Hygiene"],
    }
    assert rows[0]["source_type"] == "notes"


def test_run_batch_dry_run_writes_nothing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sink = CollectingSink()
    rows = run_graph_batch(ENTRIES, dry_run=True, sink=sink)
    assert rows and sink.rows == []


def test_run_batch_writes_through_sink(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sink = CollectingSink()
    rows = run_graph_batch(ENTRIES, dry_run=False, sink=sink)
    assert sink.rows == rows
