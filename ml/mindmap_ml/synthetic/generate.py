"""CLI: write a sample synthetic dataset to data/ for inspection.

    uv run python -m mindmap_ml.synthetic.generate [--seed N] [--days D]

The eval harness generates in-memory (no file round-trip); this is for eyeballing.
"""

from __future__ import annotations

import argparse

from ..config import DATA_DIR
from .generator import generate_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic MindMap data (fake; testing only).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=150)
    args = ap.parse_args()

    df = generate_dataset(seed=args.seed, n_days=args.days)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"synthetic_seed{args.seed}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows across {df['user_id'].nunique()} users -> {out}")
    print(df.groupby("persona")["user_id"].nunique().to_string())


if __name__ == "__main__":
    main()
