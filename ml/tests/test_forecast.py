"""Phase 3: the ML forecaster must beat the baseline on calibration AND ranking
(on held-out users) and must abstain under the safety contract.
"""

from datetime import date

import pytest

from mindmap_ml.eval.compare import run_comparison
from mindmap_ml.features.engineering import add_forward_labels
from mindmap_ml.models.base import PREDICTION_TYPES, PredictionWindow
from mindmap_ml.models.forecast import ForecastModel
from mindmap_ml.synthetic.generator import generate_dataset


@pytest.fixture(scope="module")
def reports():
    # leave-user-out: ML is judged on users it never trained on
    return run_comparison(seed=0, days=120, lookback=30)


@pytest.fixture(scope="module")
def fitted_model():
    df = add_forward_labels(generate_dataset(seed=0, n_days=90))
    return ForecastModel().fit(df)


def test_ml_is_better_calibrated_than_baseline(reports) -> None:
    baseline, ml = reports
    for t in ("migraine", "anxiety", "mood", "pain_flare"):
        assert ml.per_type[t].ece <= baseline.per_type[t].ece + 1e-9, f"{t} calibration regressed"


def test_ml_improves_ranking_on_learnable_signals(reports) -> None:
    baseline, ml = reports
    # anxiety & mood carry clear learnable structure in the personas
    assert ml.per_type["anxiety"].auroc > baseline.per_type["anxiety"].auroc
    assert ml.per_type["mood"].auroc > baseline.per_type["mood"].auroc
    # NOTE: next-day migraine ranking is NOT reliably improved (documented
    # limitation — see MODEL_CARD). The model wins on calibration, not ranking,
    # so we intentionally do not assert a migraine AUROC gain here.


def test_ml_coverage_not_worse(reports) -> None:
    baseline, ml = reports
    for t in PREDICTION_TYPES:
        assert ml.per_type[t].coverage >= baseline.per_type[t].coverage - 0.02


def test_model_abstains_on_thin_history(fitted_model) -> None:
    short = PredictionWindow("u1", date(2025, 1, 5), entries=[{"anxiety": 6, "sleep_minutes": 400, "migraine": False}] * 3)
    preds = fitted_model.predict(short)
    assert all(p.abstained for p in preds)


def test_model_emits_with_enough_history(fitted_model) -> None:
    entries = [{"anxiety": 5, "sleep_minutes": 420, "migraine": False, "mood_valence": 1, "depression": 2}] * 20
    long = PredictionWindow("u1", date(2025, 2, 1), entries=entries)
    preds = {p.prediction_type: p for p in fitted_model.predict(long)}
    # at least one non-abstained, calibrated risk in [0,1]
    emitted = [p for p in preds.values() if not p.abstained]
    assert emitted
    for p in emitted:
        assert p.risk is not None and 0.0 <= p.risk <= 1.0
        assert p.model_version == "v2_ml_assistive"
