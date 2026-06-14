"""CLI: build a clinician summary for one user (synthetic by default).

    uv run python -m mindmap_ml.reports.run_summary [--seed N] [--user PERSONA_ID]

Demonstrates the Tier-0 deliverable end-to-end without a DB. Serving (Phase 5)
persists the same structure.
"""

from __future__ import annotations

import argparse
import json

from ..synthetic.generator import generate_dataset
from .clinician_summary import build_clinician_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a clinician summary (synthetic).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--user", type=str, default="anxiety_after_poor_sleep_00")
    args = ap.parse_args()

    df = generate_dataset(seed=args.seed, n_days=args.days)
    user_df = df[df["user_id"] == args.user].copy()
    if user_df.empty:
        users = sorted(df["user_id"].unique())
        print(f"No such user '{args.user}'. Available examples: {users[:6]} ...")
        return
    summary = build_clinician_summary(user_df)
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
