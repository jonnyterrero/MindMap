"""Phase 5: batch scoring produces valid, gated, versioned, idempotent rows;
rule fallback fills in when ML abstains; dry-run writes nothing.
"""

import pytest

from mindmap_ml.config import BASELINE_MODEL_VERSION
from mindmap_ml.features.engineering import add_forward_labels
from mindmap_ml.models.base import PREDICTION_TYPES
from mindmap_ml.models.forecast import ForecastModel
from mindmap_ml.serving.score_batch import build_prediction_rows, model_status, run_batch
from mindmap_ml.serving.supabase_io import CollectingSink
from mindmap_ml.synthetic.generator import generate_dataset

REQUIRED_KEYS = {
    "user_id", "prediction_type", "entry_date", "predicted_at", "risk_score",
    "risk_level", "confidence", "uncertainty", "contributing_factors",
    "evidence_citations", "model_version", "abstained", "abstain_reason", "source",
}
UPSERT_KEY = {"user_id", "prediction_type", "entry_date", "model_version"}


@pytest.fixture(scope="module")
def data():
    return generate_dataset(seed=0, n_days=90)


@pytest.fixture(scope="module")
def model(data):
    return ForecastModel().fit(add_forward_labels(data))


@pytest.fixture(scope="module")
def ml_rows(data, model):
    return build_prediction_rows(data, model)


@pytest.fixture(scope="module")
def rules_rows(data):
    return build_prediction_rows(data, model=None)


def _validate_rows(rows: list[dict]) -> None:
    assert rows
    users = {r["user_id"] for r in rows}
    # one row per user per type
    assert len(rows) == len(users) * len(PREDICTION_TYPES)
    for r in rows:
        assert REQUIRED_KEYS.issubset(r.keys())
        assert UPSERT_KEY.issubset(r.keys())  # idempotency key fields present
        assert r["prediction_type"] in PREDICTION_TYPES
        assert r["source"] in ("rules", "ml")
        assert isinstance(r["contributing_factors"], list)
        assert isinstance(r["evidence_citations"], list)
        assert isinstance(r["abstained"], bool)
        if r["risk_score"] is not None:
            assert 0.0 <= r["risk_score"] <= 1.0
            assert r["risk_level"] in ("low", "moderate", "high", "critical")


def test_rows_valid_with_model(ml_rows) -> None:
    _validate_rows(ml_rows)
    assert any(r["source"] == "ml" for r in ml_rows)  # ML used at least sometimes


def test_rules_only_mode(rules_rows) -> None:
    _validate_rows(rules_rows)
    assert all(r["source"] == "rules" for r in rules_rows)
    assert all(r["model_version"] == BASELINE_MODEL_VERSION for r in rules_rows)
    # rule fallback always emits — never abstains in rules_only batch
    assert all(r["abstained"] is False for r in rules_rows)


def test_non_abstained_rows_carry_evidence(ml_rows) -> None:
    emitted = [r for r in ml_rows if not r["abstained"]]
    assert emitted
    assert any(len(r["evidence_citations"]) > 0 for r in emitted)


def test_dry_run_writes_nothing(data) -> None:
    sink = CollectingSink()
    rows = run_batch(data, model=None, dry_run=True, sink=sink)
    assert rows
    assert sink.rows == []  # nothing written on dry-run


def test_real_run_upserts_to_sink(data) -> None:
    sink = CollectingSink()
    rows = run_batch(data, model=None, dry_run=False, sink=sink)
    assert sink.rows == rows


def test_model_status() -> None:
    assert model_status(None)["mode"] == "rules_only"


def test_rows_json_serializable(ml_rows) -> None:
    import json

    json.dumps(ml_rows)  # must not raise
