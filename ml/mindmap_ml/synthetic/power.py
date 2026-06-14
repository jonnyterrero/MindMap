"""Power analysis / data-sufficiency.

Answers the question that actually matters at n-of-1: *how many logged days
before a real effect of a given strength is reliably detectable at our
conservative thresholds — without firing on noise?* This is the principled
source for the in-app "not enough data yet → keep logging N more days" countdown,
and it justifies the thresholds to a clinician/acquirer.

Pure simulation against the same Pearson thresholds the live correlation engine
uses. Deterministic under ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CORR_MIN_ABS_R, CORR_MIN_SAMPLE_SIZE
from ..insights.correlations import _pearson

DEFAULT_GRID = (14, 21, 30, 45, 60, 90)


def _sim_lagged(rng: np.random.Generator, n_days: int, effect: float, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """anxiety[t] correlates with (standardized) sleep[t-1] at ≈ ``effect`` (the
    target Pearson r). effect=0 → a pure-noise (null) series. ``noise`` scales the
    residual (1.0 = the clean correlation model)."""
    sleep = rng.normal(420, 60, n_days)
    z = (sleep - sleep.mean()) / (sleep.std() or 1.0)
    e = rng.standard_normal(n_days)
    resid = noise * float(np.sqrt(max(1.0 - effect**2, 0.0)))
    anx_z = np.empty(n_days)
    anx_z[0] = e[0]
    anx_z[1:] = -effect * z[:-1] + resid * e[1:]  # less sleep -> more anxiety
    anx = np.clip(np.round(4.0 + 2.0 * anx_z), 0, 10)
    return sleep, anx


def _detect(sleep: np.ndarray, anx: np.ndarray, *, min_abs_r: float, min_sample: int) -> bool:
    """Does the lag-1 sleep→anxiety correlation clear the live engine's thresholds?"""
    x, y = sleep[:-1], anx[1:]
    if len(x) < min_sample:
        return False
    r = _pearson(x, y)
    return r is not None and abs(r) >= min_abs_r


@dataclass
class PowerPoint:
    n_days: int
    power: float  # P(detect | real effect)
    false_positive_rate: float  # P(detect | no effect)


def lagged_correlation_power(
    *,
    effect: float = 0.5,
    noise: float = 1.0,
    n_days_grid: tuple[int, ...] = DEFAULT_GRID,
    n_sims: int = 200,
    seed: int = 0,
    min_abs_r: float = CORR_MIN_ABS_R,
    min_sample: int = CORR_MIN_SAMPLE_SIZE,
) -> list[PowerPoint]:
    """Detection power + false-positive rate vs number of logged days."""
    rng = np.random.default_rng(seed)
    out: list[PowerPoint] = []
    for n_days in n_days_grid:
        hits = fps = 0
        for _ in range(n_sims):
            s, a = _sim_lagged(rng, n_days, effect, noise)
            if _detect(s, a, min_abs_r=min_abs_r, min_sample=min_sample):
                hits += 1
            s0, a0 = _sim_lagged(rng, n_days, 0.0, noise)
            if _detect(s0, a0, min_abs_r=min_abs_r, min_sample=min_sample):
                fps += 1
        out.append(PowerPoint(n_days, round(hits / n_sims, 3), round(fps / n_sims, 3)))
    return out


def days_to_detect(points: list[PowerPoint], *, target_power: float = 0.8, max_fp: float = 0.1) -> int | None:
    """Smallest grid point meeting target power with acceptable false positives."""
    for p in sorted(points, key=lambda x: x.n_days):
        if p.power >= target_power and p.false_positive_rate <= max_fp:
            return p.n_days
    return None


def recommend_min_days(*, effect: float = 0.5, noise: float = 1.0, seed: int = 0) -> int:
    """A single number for the readiness countdown (falls back to the largest grid
    point if the effect isn't reliably detectable even there)."""
    pts = lagged_correlation_power(effect=effect, noise=noise, seed=seed)
    return days_to_detect(pts) or max(p.n_days for p in pts)
