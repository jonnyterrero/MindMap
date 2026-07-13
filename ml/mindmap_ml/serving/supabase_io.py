"""All Supabase I/O lives here (the one DB boundary).

Reads entries, upserts predictions. The service-role key is read from the
environment only — never hardcoded, never logged. The ``supabase`` client is an
optional dependency (``pip install 'mindmap-ml[serving]'``) and imported lazily,
so the rest of the package (and its tests) never require it.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

import pandas as pd

ENTRIES_TABLE = "mindmap_entries"
PREDICTIONS_TABLE = "mindmap_predictions"
UPSERT_CONFLICT = "user_id,prediction_type,entry_date,model_version"
SUMMARIES_TABLE = "mindmap_ml_summaries"
SUMMARIES_CONFLICT = "user_id,period_end,model_version"
JOURNAL_TABLE = "mindmap_journal_entries"
GRAPHS_TABLE = "mindmap_graphs"
GRAPHS_CONFLICT = "user_id,doc_id,pipeline_version"


class PredictionSink(Protocol):
    def upsert(self, rows: list[dict[str, Any]]) -> int: ...


def get_client() -> Any:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError(
            "Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in environment (see .env.example)."
        )
    try:
        from supabase import create_client
    except ImportError as e:  # pragma: no cover - optional dep
        raise RuntimeError("supabase client not installed — `uv pip install 'mindmap-ml[serving]'`") from e
    return create_client(url, key)


def read_entries(client: Any) -> pd.DataFrame:
    """Read the entries the model needs. The caller owns auth; this uses whatever
    client (service-role) is passed."""
    res = client.table(ENTRIES_TABLE).select("*").execute()
    df = pd.DataFrame(res.data or [])
    if not df.empty and "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.date
        if "migraine" in df.columns:
            df["migraine"] = df["migraine"].fillna(False).astype(bool)
    return df


class SupabaseSink:
    def __init__(self, client: Any) -> None:
        self.client = client

    def upsert(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        self.client.table(PREDICTIONS_TABLE).upsert(rows, on_conflict=UPSERT_CONFLICT).execute()
        return len(rows)


def read_journal_entries(client: Any) -> list[dict[str, Any]]:
    """Plaintext journal rows the graph pipeline digests. Rows already migrated
    to encrypted bodies have empty ``content`` and are skipped by the batch."""
    res = (
        client.table(JOURNAL_TABLE)
        .select("id,user_id,entry_date,title,content,created_at")
        .execute()
    )
    return list(res.data or [])


class SupabaseGraphsSink:
    def __init__(self, client: Any) -> None:
        self.client = client

    def upsert(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        self.client.table(GRAPHS_TABLE).upsert(rows, on_conflict=GRAPHS_CONFLICT).execute()
        return len(rows)


class SupabaseSummariesSink:
    def __init__(self, client: Any) -> None:
        self.client = client

    def upsert(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        self.client.table(SUMMARIES_TABLE).upsert(rows, on_conflict=SUMMARIES_CONFLICT).execute()
        return len(rows)


class CollectingSink:
    """Test/dry-run sink — records rows instead of writing them."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def upsert(self, rows: list[dict[str, Any]]) -> int:
        self.rows.extend(rows)
        return len(rows)
