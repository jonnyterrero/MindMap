"""ADR-002: read_journal_entries endpoint path + direct-read fallback.

Covers source selection (endpoint vs direct table), the `since` watermark
wiring, blank/soft-deleted filtering, and clear errors on a bad endpoint.
No network: urllib is monkeypatched.
"""

from __future__ import annotations

import json
import urllib.error
from contextlib import contextmanager
from datetime import date

import pytest

from mindmap_ml.serving import supabase_io


class _FakeQuery:
    """Minimal PostgREST-builder stand-in: records calls, returns canned rows."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.gte_args: tuple | None = None

    def select(self, _columns: str) -> _FakeQuery:
        return self

    def is_(self, _col: str, _val: str) -> _FakeQuery:
        return self

    def gte(self, col: str, val: str) -> _FakeQuery:
        self.gte_args = (col, val)
        return self

    def execute(self):  # noqa: ANN201 - mimics supabase response object
        return type("Res", (), {"data": self._rows})()


class _FakeClient:
    def __init__(self, rows: list[dict]) -> None:
        self.query = _FakeQuery(rows)

    def table(self, _name: str) -> _FakeQuery:
        return self.query


@contextmanager
def _fake_response(body: dict):
    class _Resp:
        def read(self) -> bytes:
            return json.dumps(body).encode("utf-8")

    yield _Resp()


def test_endpoint_is_used_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("ML_JOURNAL_DECRYPT_URL", "https://app.example.com/api/internal/ml/journal-entries")
    monkeypatch.setenv("ML_JOURNAL_DECRYPT_SECRET", "s3cret")

    captured: dict = {}

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        captured["url"] = req.full_url
        captured["auth"] = req.get_header("Authorization")
        return _fake_response(
            {
                "count": 2,
                "entries": [
                    {"id": "1", "user_id": "u", "entry_date": "2026-01-01", "content": "decrypted one"},
                    {"id": "2", "user_id": "u", "entry_date": "2026-01-02", "content": "   "},
                ],
            }
        )

    monkeypatch.setattr(supabase_io.urllib.request, "urlopen", fake_urlopen)

    # client must NOT be touched when the endpoint is configured.
    df = supabase_io.read_journal_entries(client=None, since="2026-01-01T00:00:00+00:00")

    assert captured["auth"] == "Bearer s3cret"
    assert "since=2026-01-01T00%3A00%3A00%2B00%3A00" in captured["url"]
    # blank-content row dropped; entry_date parsed to date.
    assert list(df["id"]) == ["1"]
    assert df.iloc[0]["entry_date"] == date(2026, 1, 1)


def test_direct_read_fallback_skips_encrypted_rows(monkeypatch) -> None:
    monkeypatch.delenv("ML_JOURNAL_DECRYPT_URL", raising=False)
    monkeypatch.delenv("ML_JOURNAL_DECRYPT_SECRET", raising=False)

    client = _FakeClient(
        [
            {"id": "1", "user_id": "u", "entry_date": "2026-01-01", "content": "plaintext", "deleted_at": None},
            {"id": "2", "user_id": "u", "entry_date": "2026-01-02", "content": None, "deleted_at": None},
        ]
    )
    df = supabase_io.read_journal_entries(client, since="2026-01-01T00:00:00+00:00")

    # encrypted-only row (content None) dropped; since forwarded to the query.
    assert list(df["id"]) == ["1"]
    assert client.query.gte_args == ("updated_at", "2026-01-01T00:00:00+00:00")
    assert "deleted_at" not in df.columns


def test_bad_endpoint_raises_clear_error(monkeypatch) -> None:
    monkeypatch.setenv("ML_JOURNAL_DECRYPT_URL", "https://app.example.com/x")
    monkeypatch.setenv("ML_JOURNAL_DECRYPT_SECRET", "nope")

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(supabase_io.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="HTTP 401"):
        supabase_io.read_journal_entries(client=None)
