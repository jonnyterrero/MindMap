import json

import pandas as pd
import pytest

from mindmap_ml.config import TIER0_SUMMARY_VERSION
from mindmap_ml.serving.summary_batch import build_summary_rows, run_summary_batch
from mindmap_ml.serving.supabase_io import CollectingSink
from mindmap_ml.synthetic.generator import generate_dataset

REQUIRED = {"user_id", "period_start", "period_end", "abstained", "payload", "model_version", "source"}
UPSERT_KEY = {"user_id", "period_end", "model_version"}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    # a few users is enough to validate the batch; keeps the slow summary build fast
    full = generate_dataset(seed=0, n_days=60)
    users = sorted(full["user_id"].unique())[:4]
    return full[full["user_id"].isin(users)].copy()


@pytest.fixture(scope="module")
def rows(data) -> list[dict]:
    return build_summary_rows(data)


def test_one_row_per_user_with_required_shape(rows, data) -> None:
    assert len(rows) == data["user_id"].nunique()
    for r in rows:
        assert REQUIRED.issubset(r)
        assert UPSERT_KEY.issubset(r)
        assert isinstance(r["abstained"], bool)
        assert r["model_version"] == TIER0_SUMMARY_VERSION
        assert r["payload"]["user_id"] == r["user_id"]


def test_rows_json_serializable(rows) -> None:
    json.dumps(rows)


def test_dry_run_writes_nothing(data) -> None:
    sink = CollectingSink()
    out = run_summary_batch(data, dry_run=True, sink=sink)
    assert out and sink.rows == []


def test_real_run_upserts(data) -> None:
    sink = CollectingSink()
    out = run_summary_batch(data, dry_run=False, sink=sink)
    assert sink.rows == out


def test_empty_entries() -> None:
    assert build_summary_rows(pd.DataFrame()) == []
