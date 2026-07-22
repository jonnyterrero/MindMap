"""Trained confidence calibrator for verifier output (HANDOFF item #4).

Recalibrates the rule_v0 confidence of surfaced claims against the gold set:
x = rule_v0 calibrated confidence, y = human verdict (supported or not).
Two monotone methods are fit — Platt scaling (2-parameter logistic) and
isotonic regression (pool-adjacent-violators) — and selected by
**leave-one-case-out cross-validation** on Brier score. With a gold set this
small, in-sample isotonic is guaranteed overfit, so LOO is the only honest
selector, and the winner must BEAT rule_v0's LOO Brier to ship at all
(mirroring the forecaster's "must beat Tier-0" gate).

The fitted artifact is committed next to the code (like evidence/seed/) so a
clean checkout — e.g. the daily GitHub Actions batch — uses the exact reviewed
parameters. Refit + re-review whenever the gold set changes:

    uv run python -m mindmap_ml.graph.calibrate          # report only
    uv run python -m mindmap_ml.graph.calibrate --write  # save if it beats rule_v0

Note: `graph/evaluate` numbers are in-sample once an artifact is active (the
artifact was fit on that same gold); the LOO table printed here is the honest
comparison.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

CALIBRATOR_PATH = Path(__file__).resolve().parent / "artifacts" / "graph_calibrator.json"
MIN_POINTS = 8  # refuse to fit on fewer surfaced gold claims


# --------------------------------------------------------------------------- #
# Data collection (gold -> (x, y) points)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CalPoint:
    case_id: str
    x: float  # rule_v0 calibrated confidence of a surfaced claim
    y: float  # 1.0 if the claim was actually supported, else 0.0


def collect_points(cases=None, entailment=None) -> list[CalPoint]:
    """Run the verifier over the gold set with rule_v0 confidence and pair each
    surfaced claim's confidence with its human verdict."""
    # Local imports: keep this module import-light so verify.py can import the
    # artifact loader below without a cycle.
    from .evaluate import _candidate_from_gold
    from .gold import GOLD_CASES
    from .ingest import digest
    from .verify import verify_graph

    if os.environ.get("MINDMAP_GRAPH_CALIBRATOR") != "off":
        # Points must reflect RULE confidence, not an already-trained artifact.
        raise RuntimeError(
            "collect_points requires MINDMAP_GRAPH_CALIBRATOR=off so x-values "
            "are rule_v0 outputs (set by the CLI automatically)."
        )

    cases = cases if cases is not None else GOLD_CASES
    points: list[CalPoint] = []
    for case in cases:
        doc, spans = digest(case.text, user_id="gold")
        art = verify_graph(doc, spans, _candidate_from_gold(doc, spans, case), entailment=entailment)
        surfaced_nodes = {n.node_id: n for n in art.nodes}
        surfaced_edges = {e.edge_id: e for e in art.edges}
        for i, claim in enumerate(case.claims):
            node_el = surfaced_nodes.get(f"nd_{doc.doc_id}_{i}")
            if node_el is not None and node_el.confidence is not None:
                points.append(CalPoint(case.case_id, node_el.confidence.calibrated, 1.0 if claim.supported else 0.0))
        for j, ge in enumerate(case.edges):
            edge_el = surfaced_edges.get(f"ed_{doc.doc_id}_{j}")
            if edge_el is not None and edge_el.confidence is not None:
                points.append(CalPoint(case.case_id, edge_el.confidence.calibrated, 1.0 if ge.supported else 0.0))
    return points


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_platt(xs: list[float], ys: list[float], l2: float = 1e-2, iters: int = 500, lr: float = 0.5) -> tuple[float, float]:
    """2-parameter logistic p = sigmoid(a*x + b), gradient descent on
    L2-regularized log-loss. Tiny n and near-separable folds are expected;
    the ridge term keeps (a, b) finite."""
    a, b = 1.0, 0.0
    n = len(xs)
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xs, ys, strict=True):
            err = _sigmoid(a * x + b) - y
            ga += err * x
            gb += err
        ga = ga / n + l2 * a
        gb = gb / n + l2 * b
        a -= lr * ga
        b -= lr * gb
    return a, b


def fit_isotonic(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    """Pool-adjacent-violators. Returns (x, fitted_p) breakpoints, x ascending."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    sx = [xs[i] for i in order]
    sy = [ys[i] for i in order]
    # blocks of (weight, mean)
    blocks: list[list[float]] = []  # [weight, mean, x_last]
    for x, y in zip(sx, sy, strict=True):
        blocks.append([1.0, y, x])
        while len(blocks) > 1 and blocks[-2][1] >= blocks[-1][1]:
            w2, m2, x2 = blocks.pop()
            w1, m1, _x1 = blocks.pop()
            blocks.append([w1 + w2, (w1 * m1 + w2 * m2) / (w1 + w2), x2])
    return [(b[2], b[1]) for b in blocks]


def _isotonic_predict(breakpoints: list[tuple[float, float]], x: float) -> float:
    p = breakpoints[0][1]
    for bx, bp in breakpoints:
        if x >= bx:
            p = bp
        else:
            break
    return p


# --------------------------------------------------------------------------- #
# Selection: leave-one-case-out Brier
# --------------------------------------------------------------------------- #
def _brier(preds: list[float], ys: list[float]) -> float:
    return sum((p - y) ** 2 for p, y in zip(preds, ys, strict=True)) / len(preds)


def loo_brier(points: list[CalPoint], method: str) -> float:
    """LOO grouped by case (claims within a case are not independent)."""
    case_ids = sorted({p.case_id for p in points})
    preds: list[float] = []
    ys: list[float] = []
    for held in case_ids:
        train = [p for p in points if p.case_id != held]
        test = [p for p in points if p.case_id == held]
        if not train or not test:
            continue
        txs, tys = [p.x for p in train], [p.y for p in train]
        for p in test:
            if method == "identity":
                preds.append(p.x)
            elif method == "platt":
                a, b = fit_platt(txs, tys)
                preds.append(_sigmoid(a * p.x + b))
            elif method == "isotonic":
                preds.append(_isotonic_predict(fit_isotonic(txs, tys), p.x))
            else:
                raise ValueError(f"unknown method {method!r}")
            ys.append(p.y)
    return _brier(preds, ys)


@dataclass
class FitReport:
    n_points: int
    n_cases: int
    loo_brier: dict[str, float]
    winner: str  # method with best LOO Brier ("identity" = keep rule_v0)
    params: dict | None  # winner's params fit on ALL points (None for identity)


def fit_and_select(points: list[CalPoint] | None = None) -> FitReport:
    points = points if points is not None else collect_points()
    if len(points) < MIN_POINTS:
        raise RuntimeError(f"only {len(points)} gold points; need >= {MIN_POINTS} to calibrate")
    loo = {m: round(loo_brier(points, m), 4) for m in ("identity", "platt", "isotonic")}
    winner = min(loo, key=lambda m: loo[m])
    params: dict | None = None
    xs, ys = [p.x for p in points], [p.y for p in points]
    if winner == "platt":
        a, b = fit_platt(xs, ys)
        params = {"a": round(a, 6), "b": round(b, 6)}
    elif winner == "isotonic":
        params = {"breakpoints": [[round(x, 4), round(p, 4)] for x, p in fit_isotonic(xs, ys)]}
    return FitReport(len(points), len({p.case_id for p in points}), loo, winner, params)


# --------------------------------------------------------------------------- #
# Artifact IO + the seam verify.py uses
# --------------------------------------------------------------------------- #
class TrainedCalibrator:
    def __init__(self, artifact: dict) -> None:
        self.method: str = artifact["method"]
        self.params: dict = artifact["params"]
        self.version: str = artifact["version"]

    def __call__(self, rule_calibrated: float) -> float:
        if self.method == "platt":
            return _sigmoid(self.params["a"] * rule_calibrated + self.params["b"])
        if self.method == "isotonic":
            bps = [(float(x), float(p)) for x, p in self.params["breakpoints"]]
            return _isotonic_predict(bps, rule_calibrated)
        raise ValueError(f"unknown method {self.method!r}")


def save_artifact(report: FitReport, path: Path = CALIBRATOR_PATH) -> dict:
    if report.winner == "identity" or report.params is None:
        raise RuntimeError("winner is rule_v0 (identity) — nothing to save")
    artifact = {
        "method": report.winner,
        "params": report.params,
        "version": f"trained_v1:{report.winner}",
        "n_points": report.n_points,
        "n_cases": report.n_cases,
        "loo_brier": report.loo_brier,
        "created_at": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


@lru_cache(maxsize=1)
def active_calibrator(path: Path = CALIBRATOR_PATH) -> TrainedCalibrator | None:
    """The calibrator verify.py applies on top of rule_v0 (None = rule_v0 only).
    Set MINDMAP_GRAPH_CALIBRATOR=off to force rule_v0."""
    if os.environ.get("MINDMAP_GRAPH_CALIBRATOR") == "off":
        return None
    if not path.exists():
        return None
    try:
        return TrainedCalibrator(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, KeyError):
        return None  # malformed artifact -> fail closed to rule_v0


def main() -> None:
    import sys

    os.environ["MINDMAP_GRAPH_CALIBRATOR"] = "off"  # x-values must be rule_v0
    active_calibrator.cache_clear()
    report = fit_and_select()
    print(f"Calibrator fit — {report.n_points} points / {report.n_cases} gold cases")
    print("  LOO Brier (lower is better):")
    for m, v in report.loo_brier.items():
        marker = "  <- winner" if m == report.winner else ""
        name = "rule_v0 (identity)" if m == "identity" else m
        print(f"    {name:<20} {v}{marker}")
    if report.winner == "identity":
        print("  rule_v0 stands — no artifact written.")
        return
    if "--write" in sys.argv:
        artifact = save_artifact(report)
        print(f"  wrote {CALIBRATOR_PATH.name}: {artifact['version']} params={artifact['params']}")
    else:
        print(f"  winner beats rule_v0 — rerun with --write to save {CALIBRATOR_PATH.name}")


if __name__ == "__main__":
    main()
