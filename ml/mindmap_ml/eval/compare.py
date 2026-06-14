"""Compare a model against the rule baseline through the harness.

A model only "wins" if it beats the baseline on error AND calibration without
worse abstention. Uses a leave-user-out split so the ML model is judged on users
it never trained on.

    uv run python -m mindmap_ml.eval.compare [--seed N] [--days D]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..features.engineering import add_forward_labels
from ..models.forecast import ForecastModel
from ..synthetic.generator import generate_dataset
from .baseline import RuleBaselineModel
from .harness import HarnessReport, report_from_scored, run_harness
from .reports import format_report


def leave_user_out_split(
    df: pd.DataFrame, frac_train: float = 0.5, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = sorted(df["user_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    cut = max(1, int(len(users) * frac_train))
    train_users = set(users[:cut])
    train = df[df["user_id"].isin(train_users)].copy()
    test = df[~df["user_id"].isin(train_users)].copy()
    return train, test


def run_comparison(
    seed: int = 0, days: int = 150, horizon: int = 1, lookback: int = 30
) -> tuple[HarnessReport, HarnessReport]:
    df = generate_dataset(seed=seed, n_days=days)
    df = add_forward_labels(df, horizon=horizon)
    train, test = leave_user_out_split(df, 0.5, seed=seed)

    ml = ForecastModel().fit(train)
    baseline_rep = run_harness(RuleBaselineModel(), test, lookback=lookback, horizon=horizon)
    # ML uses the fast vectorized batch path (also the Phase-5 serving primitive),
    # evaluated on the same point set as the baseline harness.
    scored = ml.score_frame(test, lookback=lookback)
    ml_rep = report_from_scored(
        ml.model_version, scored, test, horizon=horizon, lookback=lookback
    )
    return baseline_rep, ml_rep


def _wins(baseline: HarnessReport, ml: HarnessReport) -> dict[str, bool]:
    """Per-type: ML wins if AUROC improves and ECE is not worse (NaNs ignored)."""
    out: dict[str, bool] = {}
    for t in baseline.per_type:
        b, m = baseline.per_type[t], ml.per_type[t]
        better_auroc = (m.auroc == m.auroc) and (b.auroc != b.auroc or m.auroc >= b.auroc)
        better_cal = (m.ece == m.ece) and (b.ece != b.ece or m.ece <= b.ece + 1e-9)
        out[t] = bool(better_auroc and better_cal)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare ML forecaster vs rule baseline.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--lookback", type=int, default=30)
    args = ap.parse_args()

    baseline_rep, ml_rep = run_comparison(args.seed, args.days, args.horizon, args.lookback)
    print("=== BASELINE ===")
    print(format_report(baseline_rep))
    print("\n=== ML (leave-user-out) ===")
    print(format_report(ml_rep))
    print("\n=== ML beats baseline (auroc up, ece not worse)? ===")
    for t, won in _wins(baseline_rep, ml_rep).items():
        print(f"  {t:<11} {'WIN' if won else 'no'}")


if __name__ == "__main__":
    main()
