"""CLI: run the rule baseline over synthetic data and emit a report.

    uv run python -m mindmap_ml.eval.run [--seed N] [--days D] [--horizon H]

Phase 1 deliverable: proves the harness emits a calibration + abstention report
on synthetic data. Phase 3 will add `--model` to compare ML vs. baseline.
"""

from __future__ import annotations

import argparse

from ..features.engineering import add_forward_labels
from ..synthetic.generator import generate_dataset
from .baseline import RuleBaselineModel
from .harness import run_harness
from .reports import format_report, write_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline harness on synthetic data.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--no-abstain", action="store_true", help="disable abstention (fallback mode)")
    args = ap.parse_args()

    df = generate_dataset(seed=args.seed, n_days=args.days)
    df = add_forward_labels(df, horizon=args.horizon)

    model = RuleBaselineModel(apply_abstention=not args.no_abstain)
    report = run_harness(model, df, lookback=args.lookback, horizon=args.horizon)

    path = write_report(report)
    print(format_report(report))
    print(f"\nWrote JSON report -> {path}")


if __name__ == "__main__":
    main()
