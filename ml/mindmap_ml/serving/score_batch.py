"""Daily batch scoring: entries -> features -> models -> safety gate -> rows.

Scores the latest as-of day per user. ML predictions are used when available and
the model did not abstain; otherwise it **falls back to the rule engine** (which
always emits a conservative estimate). Every persisted row carries a
``model_version`` and ``source`` (rules|ml), passes contributing-factor text
through the output gate, and (when not abstaining) attaches evidence citations.

Idempotent: rows upsert on (user_id, prediction_type, entry_date, model_version).
Dry-run prints rows without writing. The Next app only reads these rows.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from typing import Any, cast

import pandas as pd

from ..config import BASELINE_MODEL_VERSION
from ..eval.baseline import RuleBaselineModel, _level_for
from ..eval.harness import _clean_entry, _window_inputs
from ..evidence.retrieve import evidence_for
from ..models.base import PREDICTION_TYPES, PredictionWindow
from ..safety.gate import check_output

USER_COL = "user_id"
DATE_COL = "entry_date"

# Map each prediction type to the curated-prior factor used to ground its row.
TYPE_FACTOR = {
    "migraine": "sleep_deficit",
    "anxiety": "sleep_deficit",
    "mood": "routine_disruption",
    "pain_flare": "sleep_deficit",
}


def model_status(model: Any | None = None) -> dict[str, Any]:
    """Current operating mode — exposed for an /api/.../model-status endpoint."""
    if model is None:
        return {"mode": "rules_only", "model_version": BASELINE_MODEL_VERSION, "abstention": "enabled"}
    return {"mode": "ml_assistive", "model_version": model.model_version, "abstention": "enabled"}


def _gate_factors(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run each factor's free-text detail through the output gate (defensive —
    these are benign by construction, but nothing user-facing skips the gate)."""
    safe: list[dict[str, Any]] = []
    for f in factors:
        detail = f.get("detail")
        if isinstance(detail, str):
            res = check_output(detail, is_risk_claim=False)
            if not res.allowed:
                f = {**f, "detail": res.safe_text}
        safe.append(f)
    return safe


def _evidence_citations(ptype: str) -> list[str]:
    factor = TYPE_FACTOR.get(ptype)
    if not factor:
        return []
    return evidence_for(factor, ptype).citations


def _row(
    *,
    user_id: str,
    entry_date: date,
    ptype: str,
    risk: float | None,
    confidence: float,
    source: str,
    model_version: str,
    abstained: bool,
    abstain_reason: str | None,
    factors: list[dict[str, Any]],
    attach_evidence: bool,
    now_iso: str,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "prediction_type": ptype,
        "entry_date": entry_date.isoformat() if isinstance(entry_date, date) else str(entry_date),
        "predicted_at": now_iso,
        "risk_score": None if risk is None else round(float(risk), 3),
        "risk_level": None if risk is None else _level_for(float(risk)),
        "confidence": round(float(confidence), 3),
        "uncertainty": round(1.0 - float(confidence), 3),
        "contributing_factors": _gate_factors(factors),
        "evidence_citations": (_evidence_citations(ptype) if (attach_evidence and not abstained) else []),
        "model_version": model_version,
        "abstained": bool(abstained),
        "abstain_reason": abstain_reason,
        "source": source,
    }


def build_prediction_rows(
    entries: pd.DataFrame,
    model: Any | None = None,
    *,
    attach_evidence: bool = True,
) -> list[dict[str, Any]]:
    """Pure: build gated, versioned prediction rows for the latest day per user.

    ``model`` is any object with ``score_frame`` + ``model_version`` (a fitted
    :class:`ForecastModel`). When None, runs in rules_only mode.
    """
    if entries.empty:
        return []
    df = entries.sort_values([USER_COL, DATE_COL]).reset_index(drop=True)
    now_iso = datetime.now(UTC).isoformat()
    rule_model = RuleBaselineModel(apply_abstention=False)  # fallback always emits

    scored_last = None
    if model is not None:
        scored = model.score_frame(df)
        scored_last = scored.groupby(USER_COL, sort=False).tail(1).set_index(USER_COL)

    rows: list[dict[str, Any]] = []
    for uid, g in df.groupby(USER_COL, sort=False):
        g = g.sort_values(DATE_COL)
        latest_date = g.iloc[-1][DATE_COL]
        cleaned = [_clean_entry(cast("dict[str, Any]", r)) for r in g.to_dict("records")]
        newest_first = cleaned[::-1]
        side = _window_inputs(newest_first)
        window = PredictionWindow(
            user_id=str(uid),
            as_of_date=latest_date,
            entries=newest_first,
            wearable=side["wearable"],
            weather=side["weather"],
            body_pain=side["body_pain"],
        )
        rule_preds = {p.prediction_type: p for p in rule_model.predict(window)}

        for t in PREDICTION_TYPES:
            use_ml = (
                scored_last is not None
                and uid in scored_last.index
                and not bool(scored_last.loc[uid, f"abstained_{t}"])
                and not pd.isna(scored_last.loc[uid, f"risk_{t}"])
            )
            if use_ml:
                assert scored_last is not None and model is not None  # narrowed by use_ml
                rows.append(
                    _row(
                        user_id=str(uid),
                        entry_date=latest_date,
                        ptype=t,
                        risk=float(scored_last.loc[uid, f"risk_{t}"]),
                        confidence=float(scored_last.loc[uid, f"confidence_{t}"]),
                        source="ml",
                        model_version=model.model_version,
                        abstained=False,
                        abstain_reason=None,
                        factors=[],
                        attach_evidence=attach_evidence,
                        now_iso=now_iso,
                    )
                )
            else:
                rp = rule_preds[t]
                rows.append(
                    _row(
                        user_id=str(uid),
                        entry_date=latest_date,
                        ptype=t,
                        risk=rp.risk,
                        confidence=rp.confidence,
                        source="rules",
                        model_version=rule_model.model_version,
                        abstained=rp.abstained,
                        abstain_reason=rp.abstain_reason,
                        factors=[f.to_dict() for f in rp.contributing_factors],
                        attach_evidence=attach_evidence,
                        now_iso=now_iso,
                    )
                )
    return rows


def run_batch(
    entries: pd.DataFrame,
    model: Any | None = None,
    *,
    dry_run: bool = True,
    sink: Any | None = None,
    attach_evidence: bool = True,
) -> list[dict[str, Any]]:
    """Build rows and (unless dry-run) upsert them via ``sink``."""
    rows = build_prediction_rows(entries, model, attach_evidence=attach_evidence)
    if dry_run or sink is None:
        return rows
    sink.upsert(rows)
    return rows


def _load_model(path: str | None) -> Any | None:
    # SECURITY: joblib uses pickle, which can execute arbitrary code on load. Only
    # ever pass a path to a model artifact WE produced via our own training and
    # stored under our control (ml/artifacts/). Never load a model file from an
    # untrusted/user-supplied source.
    if not path:
        return None
    import joblib

    return joblib.load(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="MindMap ML daily batch scoring.")
    ap.add_argument("--dry-run", action="store_true", help="print rows; do not write")
    ap.add_argument("--synthetic", action="store_true", help="score synthetic data instead of Supabase")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--model", type=str, default=None, help="path to a saved ForecastModel (joblib)")
    args = ap.parse_args()

    model = _load_model(args.model)

    if args.synthetic:
        from ..synthetic.generator import generate_dataset

        entries = generate_dataset(seed=args.seed, n_days=args.days)
        sink = None
    else:
        from .supabase_io import SupabaseSink, get_client, read_entries

        client = get_client()
        entries = read_entries(client)
        sink = None if args.dry_run else SupabaseSink(client)

    print(f"mode: {json.dumps(model_status(model))}")
    rows = run_batch(entries, model, dry_run=args.dry_run, sink=sink)
    n_abstained = sum(1 for r in rows if r["abstained"])
    print(f"built {len(rows)} rows ({n_abstained} abstained) for {entries[USER_COL].nunique()} users")
    if args.dry_run:
        for r in rows[:8]:
            print(json.dumps(r))
        print("(dry-run: nothing written)")


if __name__ == "__main__":
    main()
