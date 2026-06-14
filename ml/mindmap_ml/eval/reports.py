"""Serialize and pretty-print harness reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..config import REPORTS_DIR
from .harness import HarnessReport


def _fmt(x: float) -> str:
    return "  n/a" if x != x else f"{x:5.3f}"  # x!=x -> NaN


def write_report(report: HarnessReport, path: Path | None = None) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = path or (REPORTS_DIR / f"harness_{report.model_version}.json")
    out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return out


def format_report(report: HarnessReport) -> str:
    lines = [
        f"Harness report — model={report.model_version}",
        f"  users={report.n_users}  points={report.n_points_total}  "
        f"horizon={report.horizon}d  lookback={report.lookback}",
        "",
        f"  {'type':<11} {'n':>5} {'cov':>5} {'pos':>5} {'brier':>6} {'ece':>6} {'auroc':>6} {'auprc':>6} {'rec@.5':>6}",
    ]
    for t, r in report.per_type.items():
        lines.append(
            f"  {t:<11} {r.n_covered:>5} {_fmt(r.coverage)} {_fmt(r.positive_rate)} "
            f"{_fmt(r.brier)} {_fmt(r.ece)} {_fmt(r.auroc)} {_fmt(r.auprc)} {_fmt(r.recall_at_0_5)}"
        )
    lines.append("")
    lines.append("  (cov=coverage, pos=positive rate on covered, lower brier/ece = better)")
    return "\n".join(lines)
