"""Daily batch: build a Tier-0 clinician summary per user → upsert to Supabase.

The app only reads `mindmap_ml_summaries`; this is the writer. Idempotent on
(user_id, period_end, model_version). Dry-run prints without writing. Pure
``build_summary_rows`` is unit-tested; ``run_summary_batch`` wires the sink.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ..config import TIER0_SUMMARY_VERSION
from ..reports.clinician_summary import build_clinician_summary

USER_COL = "user_id"
DATE_COL = "entry_date"


def build_summary_rows(entries: pd.DataFrame, *, model_version: str = TIER0_SUMMARY_VERSION) -> list[dict[str, Any]]:
    """One clinician-summary row per user (latest as-of = max entry_date)."""
    if entries.empty:
        return []
    df = entries.sort_values([USER_COL, DATE_COL])
    now_iso = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for uid, g in df.groupby(USER_COL, sort=False):
        summary = build_clinician_summary(g.copy())
        dates = sorted(g[DATE_COL].tolist())
        rows.append({
            "user_id": str(uid),
            "period_start": str(dates[0]),
            "period_end": str(dates[-1]),
            "abstained": bool(summary.abstained),
            "payload": summary.to_dict(),
            "model_version": model_version,
            "source": "rules",
            "updated_at": now_iso,
        })
    return rows


def run_summary_batch(
    entries: pd.DataFrame, *, dry_run: bool = True, sink: Any | None = None
) -> list[dict[str, Any]]:
    rows = build_summary_rows(entries)
    if dry_run or sink is None:
        return rows
    sink.upsert(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="MindMap Tier-0 clinician-summary batch.")
    ap.add_argument("--dry-run", action="store_true", help="print rows; do not write")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data instead of Supabase")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=150)
    args = ap.parse_args()

    if args.synthetic:
        from ..synthetic.generator import generate_dataset

        entries = generate_dataset(seed=args.seed, n_days=args.days)
        sink = None
    else:
        from .supabase_io import SupabaseSummariesSink, get_client, read_entries

        client = get_client()
        entries = read_entries(client)
        sink = None if args.dry_run else SupabaseSummariesSink(client)

    rows = run_summary_batch(entries, dry_run=args.dry_run, sink=sink)
    n_abstained = sum(1 for r in rows if r["abstained"])
    n_users = entries[USER_COL].nunique() if not entries.empty else 0
    print(f"built {len(rows)} summaries ({n_abstained} abstained) for {n_users} users")
    if args.dry_run:
        for r in rows[:3]:
            print(json.dumps(r)[:600])
        print("(dry-run: nothing written)")


if __name__ == "__main__":
    main()
