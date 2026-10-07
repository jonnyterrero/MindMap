"""All Supabase I/O lives here (the one DB boundary).

Reads entries, upserts predictions. The service-role key is read from the
environment only — never hardcoded, never logged. The ``supabase`` client is an
optional dependency (``pip install 'mindmap-ml[serving]'``) and imported lazily,
so the rest of the package (and its tests) never require it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

import pandas as pd

ENTRIES_TABLE = "mindmap_entries"
PREDICTIONS_TABLE = "mindmap_predictions"
UPSERT_CONFLICT = "user_id,prediction_type,entry_date,model_version"
SUMMARIES_TABLE = "mindmap_ml_summaries"
SUMMARIES_CONFLICT = "user_id,period_end,model_version"
JOURNAL_TABLE = "mindmap_journal_entries"
GRAPHS_TABLE = "mindmap_graphs"
GRAPHS_CONFLICT = "user_id,source_table,source_id,pipeline_version"


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


def _fetch_decrypted_journal(url: str, secret: str, since: str | None) -> list[dict[str, Any]]:
    """Pull decrypted journal entries from the Vercel decrypt endpoint (ADR-002).

    The master key lives only in Vercel, so encrypted bodies are decrypted there
    and returned over TLS. Plaintext stays in memory here and is never persisted
    beyond the quoted spans the graph already stores. Uses stdlib urllib so the
    batch gains no new dependency.
    """
    if since:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}since={urllib.parse.quote(since)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {secret}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (fixed https endpoint)
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"journal decrypt endpoint returned HTTP {e.code} (check ML_JOURNAL_DECRYPT_SECRET / URL)"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"journal decrypt endpoint unreachable: {e.reason}") from e
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise RuntimeError("journal decrypt endpoint returned a malformed payload")
    return entries


def read_journal_entries(client: Any, since: str | None = None) -> pd.DataFrame:
    """Read journal entries the graph pipeline runs over.

    Two sources, chosen by environment:

    * **Decrypt endpoint** (when ``ML_JOURNAL_DECRYPT_URL`` +
      ``ML_JOURNAL_DECRYPT_SECRET`` are set): the Vercel route decrypts
      encrypted bodies and returns plaintext, so the pipeline covers encrypted
      entries without the master key ever reaching this (GitHub Actions)
      environment. See ADR-002.
    * **Direct table read** (fallback, e.g. local/dev without the endpoint):
      selects only the fields the writer needs so encrypted blobs never leave
      the DB. Rows encrypted-only (plaintext ``content`` is null) are skipped,
      exactly as before the endpoint existed.

    ``since`` is an optional ISO-8601 watermark (``updated_at >= since``) that
    bounds how much plaintext a run surfaces; the caller's ``content_sha`` skip
    still prevents recomputing unchanged entries, so a watermark never drops
    work. Soft-deleted rows and blank content are always dropped.
    """
    url = os.environ.get("ML_JOURNAL_DECRYPT_URL")
    secret = os.environ.get("ML_JOURNAL_DECRYPT_SECRET")

    if url and secret:
        df = pd.DataFrame(_fetch_decrypted_journal(url, secret, since))
    else:
        query = (
            client.table(JOURNAL_TABLE)
            .select("id, user_id, entry_date, content, deleted_at")
            .is_("deleted_at", "null")
        )
        if since:
            query = query.gte("updated_at", since)
        res = query.execute()
        df = pd.DataFrame(res.data or [])

    if df.empty:
        return df
    df = df[df["content"].notna() & (df["content"].str.strip() != "")]
    if "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.date
    return df.drop(columns=["deleted_at"], errors="ignore").reset_index(drop=True)


class SupabaseSink:
    def __init__(self, client: Any) -> None:
        self.client = client

    def upsert(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        self.client.table(PREDICTIONS_TABLE).upsert(rows, on_conflict=UPSERT_CONFLICT).execute()
        return len(rows)


class SupabaseSummariesSink:
    def __init__(self, client: Any) -> None:
        self.client = client

    def upsert(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        self.client.table(SUMMARIES_TABLE).upsert(rows, on_conflict=SUMMARIES_CONFLICT).execute()
        return len(rows)


def read_graph_shas(client: Any) -> dict[str, str]:
    """Map ``source_id -> content_sha`` for already-built graphs, so the batch
    can skip journal entries whose text hasn't changed."""
    res = client.table(GRAPHS_TABLE).select("source_id, content_sha").execute()
    return {str(r["source_id"]): r["content_sha"] for r in (res.data or [])}


class SupabaseGraphSink:
    def __init__(self, client: Any) -> None:
        self.client = client

    def upsert(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        self.client.table(GRAPHS_TABLE).upsert(rows, on_conflict=GRAPHS_CONFLICT).execute()
        return len(rows)


class CollectingSink:
    """Test/dry-run sink — records rows instead of writing them."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def upsert(self, rows: list[dict[str, Any]]) -> int:
        self.rows.extend(rows)
        return len(rows)
