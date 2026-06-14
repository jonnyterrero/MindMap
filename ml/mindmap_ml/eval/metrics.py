"""Evaluation metrics — pure numpy, no sklearn dependency.

Beyond error metrics, this module measures **calibration** (reliability / ECE)
and **abstention** (coverage vs. accuracy), because the safety contract judges a
model on whether its uncertainty is honest, not only on whether it's accurate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayLike = np.ndarray | list[float]


def _clean(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(s))
    return y[mask], s[mask]


def _average_ranks(sorted_vals: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged, for an already-sorted array."""
    n = len(sorted_vals)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1  # average of (i+1..j+1)
        i = j + 1
    return ranks


def brier_score(y_true: ArrayLike, y_score: ArrayLike) -> float:
    y, s = _clean(y_true, y_score)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((s - y) ** 2))


def auroc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under ROC via the rank (Mann–Whitney U) identity; ties handled."""
    y, s = _clean(y_true, y_score)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = _average_ranks(s[order])
    sum_ranks_pos = ranks[pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Average precision (area under the precision-recall curve)."""
    y, s = _clean(y_true, y_score)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_pred: float  # average predicted probability (confidence)
    mean_true: float  # observed positive rate (accuracy)


def reliability_curve(y_true: ArrayLike, y_score: ArrayLike, n_bins: int = 10) -> list[ReliabilityBin]:
    y, s = _clean(y_true, y_score)
    bins: list[ReliabilityBin] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # last bin is closed on the right so p == 1.0 lands somewhere
        in_bin = (s >= lo) & (s < hi) if b < n_bins - 1 else (s >= lo) & (s <= hi)
        cnt = int(in_bin.sum())
        if cnt == 0:
            bins.append(ReliabilityBin(lo, hi, 0, float("nan"), float("nan")))
            continue
        bins.append(
            ReliabilityBin(lo, hi, cnt, float(s[in_bin].mean()), float(y[in_bin].mean()))
        )
    return bins


def expected_calibration_error(y_true: ArrayLike, y_score: ArrayLike, n_bins: int = 10) -> float:
    y, s = _clean(y_true, y_score)
    if len(y) == 0:
        return float("nan")
    total = len(y)
    ece = 0.0
    for b in reliability_curve(y, s, n_bins):
        if b.count == 0:
            continue
        ece += (b.count / total) * abs(b.mean_true - b.mean_pred)
    return float(ece)


@dataclass
class ThresholdMetrics:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    accuracy: float


def binary_metrics_at_threshold(y_true: ArrayLike, y_score: ArrayLike, threshold: float = 0.5) -> ThresholdMetrics:
    y, s = _clean(y_true, y_score)
    yhat = (s >= threshold).astype(int)
    tp = int(np.sum((yhat == 1) & (y == 1)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    fn = int(np.sum((yhat == 0) & (y == 1)))
    tn = int(np.sum((yhat == 0) & (y == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y) if len(y) else float("nan")
    return ThresholdMetrics(threshold, tp, fp, fn, tn, precision, recall, f1, accuracy)


def coverage(abstained: ArrayLike) -> float:
    """Fraction of points the model actually scored (did not abstain on)."""
    a = np.asarray(abstained, dtype=bool)
    return float(1.0 - a.mean()) if len(a) else float("nan")
