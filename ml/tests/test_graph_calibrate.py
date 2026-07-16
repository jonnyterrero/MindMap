"""Trained calibrator: fitting math, LOO selection, artifact IO, verify seam."""

from __future__ import annotations

import json

import pytest

from mindmap_ml.graph.calibrate import (
    CalPoint,
    FitReport,
    TrainedCalibrator,
    _sigmoid,
    active_calibrator,
    collect_points,
    fit_and_select,
    fit_isotonic,
    fit_platt,
    loo_brier,
    save_artifact,
)


@pytest.fixture(autouse=True)
def _rule_v0_only(monkeypatch):
    """Force rule_v0 in verify and clear the artifact cache around each test."""
    monkeypatch.setenv("MINDMAP_GRAPH_CALIBRATOR", "off")
    active_calibrator.cache_clear()
    yield
    active_calibrator.cache_clear()


# --------------------------------------------------------------------------- #
# fitting math
# --------------------------------------------------------------------------- #
def test_platt_is_monotone_and_bounded() -> None:
    xs = [0.2, 0.4, 0.6, 0.8, 0.35, 0.9]
    ys = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    a, b = fit_platt(xs, ys)
    assert a > 0  # higher rule confidence -> higher probability
    preds = [_sigmoid(a * x + b) for x in (0.1, 0.5, 0.9)]
    assert preds == sorted(preds)
    assert all(0.0 < p < 1.0 for p in preds)


def test_isotonic_pav_pools_violators() -> None:
    # y dips at x=0.5 -> PAV must pool it with a neighbor (monotone output)
    bps = fit_isotonic([0.1, 0.5, 0.9], [0.0, 1.0, 0.5])
    fitted = [p for _, p in bps]
    assert fitted == sorted(fitted)


def test_isotonic_perfect_data_recovers_steps() -> None:
    bps = fit_isotonic([0.2, 0.4, 0.6, 0.8], [0.0, 0.0, 1.0, 1.0])
    cal = TrainedCalibrator({"method": "isotonic", "params": {"breakpoints": bps}, "version": "t"})
    assert cal(0.3) == 0.0 and cal(0.9) == 1.0


# --------------------------------------------------------------------------- #
# LOO selection
# --------------------------------------------------------------------------- #
def _toy_points() -> list[CalPoint]:
    # Systematically overconfident rule output (x ~0.85, true rate 0.5). Each
    # case mixes one supported and one unsupported claim so LOO folds keep the
    # class balance (class-pure cases would make LOO degenerate).
    pts = []
    for i, (x, y) in enumerate([(0.9, 1), (0.85, 0), (0.8, 1), (0.9, 0), (0.85, 1), (0.8, 0), (0.9, 1), (0.85, 0)]):
        pts.append(CalPoint(f"case{i // 2}", x, float(y)))
    return pts


def test_loo_groups_by_case_and_scores_all_methods() -> None:
    pts = _toy_points()
    for method in ("identity", "platt", "isotonic"):
        b = loo_brier(pts, method)
        assert 0.0 <= b <= 1.0


def test_overconfident_rule_loses_to_platt_on_loo() -> None:
    # x≈0.85 while the base rate is 0.5 -> identity Brier ≈ 0.36; any shrinkage wins
    pts = _toy_points()
    assert loo_brier(pts, "platt") < loo_brier(pts, "identity")


def test_fit_and_select_refuses_tiny_data() -> None:
    with pytest.raises(RuntimeError, match="need >="):
        fit_and_select([CalPoint("a", 0.5, 1.0)] * 3)


# --------------------------------------------------------------------------- #
# gold collection + artifact IO + verify seam
# --------------------------------------------------------------------------- #
def test_collect_points_pairs_confidence_with_verdict() -> None:
    pts = collect_points()
    assert len(pts) >= 8  # enough surfaced gold claims to calibrate on
    assert all(0.0 <= p.x <= 1.0 and p.y in (0.0, 1.0) for p in pts)
    assert any(p.y == 0.0 for p in pts)  # the measured false-accepts are present


def test_collect_points_requires_rule_v0(monkeypatch) -> None:
    monkeypatch.delenv("MINDMAP_GRAPH_CALIBRATOR", raising=False)
    with pytest.raises(RuntimeError, match="rule_v0"):
        collect_points()


def test_artifact_roundtrip_and_active_calibrator(tmp_path, monkeypatch) -> None:
    report = FitReport(
        n_points=10, n_cases=5,
        loo_brier={"identity": 0.2, "platt": 0.1, "isotonic": 0.15},
        winner="platt", params={"a": 2.0, "b": -1.0},
    )
    path = tmp_path / "cal.json"
    artifact = save_artifact(report, path)
    assert artifact["version"] == "trained_v1:platt"
    data = json.loads(path.read_text(encoding="utf-8"))
    cal = TrainedCalibrator(data)
    assert cal(0.5) == pytest.approx(_sigmoid(2.0 * 0.5 - 1.0))

    monkeypatch.delenv("MINDMAP_GRAPH_CALIBRATOR", raising=False)
    active_calibrator.cache_clear()
    loaded = active_calibrator(path)
    assert loaded is not None and loaded.version == "trained_v1:platt"


def test_identity_winner_saves_nothing(tmp_path) -> None:
    report = FitReport(8, 4, {"identity": 0.1, "platt": 0.2, "isotonic": 0.2}, "identity", None)
    with pytest.raises(RuntimeError, match="identity"):
        save_artifact(report, tmp_path / "cal.json")


def test_env_off_forces_rule_v0(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cal.json"
    save_artifact(
        FitReport(10, 5, {"identity": 0.2, "platt": 0.1, "isotonic": 0.15}, "platt", {"a": 1.0, "b": 0.0}),
        path,
    )
    monkeypatch.setenv("MINDMAP_GRAPH_CALIBRATOR", "off")
    active_calibrator.cache_clear()
    assert active_calibrator(path) is None


def test_malformed_artifact_fails_closed_to_rule_v0(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cal.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.delenv("MINDMAP_GRAPH_CALIBRATOR", raising=False)
    active_calibrator.cache_clear()
    assert active_calibrator(path) is None
