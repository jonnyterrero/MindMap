"""The harness must emit a calibration + abstention report on synthetic data."""

from mindmap_ml.eval.baseline import RuleBaselineModel
from mindmap_ml.eval.harness import run_harness
from mindmap_ml.eval.reports import format_report
from mindmap_ml.features.engineering import add_forward_labels
from mindmap_ml.models.base import PREDICTION_TYPES
from mindmap_ml.synthetic.generator import generate_dataset


def _report():
    df = generate_dataset(seed=0, n_days=60)
    df = add_forward_labels(df, horizon=1)
    return run_harness(RuleBaselineModel(), df, lookback=21, horizon=1)


def test_harness_produces_all_types() -> None:
    rep = _report()
    assert set(rep.per_type.keys()) == set(PREDICTION_TYPES)
    assert rep.n_points_total > 0


def test_each_type_has_calibration_and_abstention_fields() -> None:
    rep = _report()
    for t in PREDICTION_TYPES:
        r = rep.per_type[t]
        assert r.n_points > 0
        assert 0.0 <= r.coverage <= 1.0
        # calibration + ranking metrics exist (float; may be nan for thin classes)
        assert isinstance(r.brier, float)
        assert isinstance(r.ece, float)
        assert isinstance(r.auroc, float)


def test_format_report_renders() -> None:
    text = format_report(_report())
    assert "Harness report" in text
    assert "auroc" in text


def test_abstention_reduces_coverage_for_thin_history() -> None:
    # With abstention on, early low-history points are abstained -> coverage < 1.
    rep = _report()
    assert any(rep.per_type[t].coverage < 1.0 for t in PREDICTION_TYPES)
