import numpy as np

from mindmap_ml.eval import metrics


def test_auroc_perfect_and_inverted() -> None:
    assert metrics.auroc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.9]) == 1.0
    assert metrics.auroc([0, 1], [0.9, 0.1]) == 0.0


def test_auroc_handles_ties() -> None:
    # all scores equal -> auroc 0.5
    assert metrics.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auroc_single_class_is_nan() -> None:
    assert np.isnan(metrics.auroc([1, 1, 1], [0.2, 0.5, 0.9]))


def test_auprc_perfect() -> None:
    assert metrics.auprc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_brier() -> None:
    assert metrics.brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert metrics.brier_score([1, 0], [0.5, 0.5]) == 0.25


def test_ece_perfectly_calibrated_is_zero() -> None:
    ece = metrics.expected_calibration_error([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0], n_bins=2)
    assert ece == 0.0


def test_coverage() -> None:
    assert metrics.coverage([False, False, True, True]) == 0.5
    assert metrics.coverage([False, False]) == 1.0


def test_threshold_metrics() -> None:
    m = metrics.binary_metrics_at_threshold([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], 0.5)
    assert m.tp == 2 and m.tn == 2 and m.fp == 0 and m.fn == 0
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0
