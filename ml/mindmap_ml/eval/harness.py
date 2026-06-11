"""Evaluation harness.

Runs any :class:`Model` over a labeled daily frame at every valid (user, as-of
day) point — strictly past+current entries in, future label out (no leakage) —
and produces a standardized per-type report covering error, **calibration**, and
**abstention**. The same harness judges the rule baseline and every ML model, so
"the model wins" is a like-for-like comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..models.base import Model, PredictionWindow
from . import metrics
from .baseline import _num  # reuse the JS-faithful numeric coercion

USER_COL = "user_id"
DATE_COL = "entry_date"

LABELS = {
    "migraine": "label_migraine",
    "anxiety": "label_anxiety",
    "mood": "label_mood",
    "pain_flare": "label_pain_flare",
}

# Fields the rule baseline reads; cleaned to plain Python types so the JS-faithful
# helpers (`is True`, isinstance(int/float)) behave correctly on numpy values.
_NUMERIC_ENTRY_FIELDS = (
    "sleep_minutes",
    "sleep_quality",
    "hrv",
    "mood_valence",
    "anxiety",
    "depression",
    "mania",
    "focus",
    "productivity",
    "migraine_intensity",
    "body_pain",
    "pressure",
    "humidity",
    "temp_max",
    "pressure_change",
)


def _clean_entry(rec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in _NUMERIC_ENTRY_FIELDS:
        if k in rec:
            v = rec[k]
            out[k] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
    mig = rec.get("migraine")
    out["migraine"] = bool(mig) if not (mig is None or (isinstance(mig, float) and np.isnan(mig))) else False
    out["pollen_level"] = rec.get("pollen_level")
    out["entry_date"] = rec.get("entry_date")
    return out


def _window_inputs(entries_newest_first: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the wearable/weather/body_pain side-inputs the baseline expects
    from the most-recent entries (mirrors how the app gathers them)."""
    latest = entries_newest_first[0]
    recent7 = entries_newest_first[:7]
    body_vals = [_num(e.get("body_pain")) for e in recent7]
    body_present = [v for v in body_vals if v is not None]
    body_pain = {
        "avgIntensity": (sum(body_present) / len(body_present)) if body_present else None,
        "daysWithPain": sum(1 for v in body_present if v > 0),
    }
    return {
        "wearable": {"hrv": latest.get("hrv")},
        "weather": {
            "pressure_change": latest.get("pressure_change"),
            "pollen_level": latest.get("pollen_level"),
        },
        "body_pain": body_pain,
    }


@dataclass
class TypeReport:
    prediction_type: str
    n_points: int
    n_covered: int
    coverage: float
    positive_rate: float
    brier: float
    ece: float
    auroc: float
    auprc: float
    precision_at_0_5: float
    recall_at_0_5: float


@dataclass
class HarnessReport:
    model_version: str
    n_users: int
    n_points_total: int
    horizon: int
    lookback: int
    per_type: dict[str, TypeReport] = field(default_factory=dict)
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def run_harness(
    model: Model,
    df: pd.DataFrame,
    *,
    lookback: int = 30,
    horizon: int = 1,
    calibration_bins: int = 10,
) -> HarnessReport:
    df = df.sort_values([USER_COL, DATE_COL]).reset_index(drop=True)

    # accumulators per type
    scores: dict[str, list[float]] = {t: [] for t in LABELS}
    labels: dict[str, list[float]] = {t: [] for t in LABELS}
    abstained: dict[str, list[bool]] = {t: [] for t in LABELS}

    users = df[USER_COL].unique()
    n_points_total = 0

    for uid in users:
        urows = df[df[USER_COL] == uid].to_dict("records")
        cleaned = [_clean_entry(r) for r in urows]
        for i, raw in enumerate(urows):
            # need all labels present (drops the final `horizon` tail rows)
            label_vals = {t: raw.get(LABELS[t]) for t in LABELS}
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in label_vals.values()):
                continue

            newest_first = cleaned[: i + 1][::-1][:lookback]
            side = _window_inputs(newest_first)
            window = PredictionWindow(
                user_id=str(uid),
                as_of_date=raw[DATE_COL],
                entries=newest_first,
                wearable=side["wearable"],
                weather=side["weather"],
                body_pain=side["body_pain"],
            )
            preds = {p.prediction_type: p for p in model.predict(window)}
            n_points_total += 1
            for t in LABELS:
                p = preds.get(t)
                y_val = float(label_vals[t])
                if p is None or p.abstained or p.risk is None:
                    abstained[t].append(True)
                    scores[t].append(float("nan"))
                    labels[t].append(y_val)
                else:
                    abstained[t].append(False)
                    scores[t].append(float(p.risk))
                    labels[t].append(y_val)

    per_type: dict[str, TypeReport] = {}
    for t in LABELS:
        ab = np.asarray(abstained[t], dtype=bool)
        s = np.asarray(scores[t], dtype=float)
        y = np.asarray(labels[t], dtype=float)
        covered = ~ab
        s_cov, y_cov = s[covered], y[covered]
        n_points = int(len(y))
        n_covered = int(covered.sum())
        pos_rate = float(y_cov.mean()) if n_covered else float("nan")
        thr = metrics.binary_metrics_at_threshold(y_cov, s_cov, 0.5) if n_covered else None
        per_type[t] = TypeReport(
            prediction_type=t,
            n_points=n_points,
            n_covered=n_covered,
            coverage=metrics.coverage(ab) if n_points else float("nan"),
            positive_rate=pos_rate,
            brier=metrics.brier_score(y_cov, s_cov) if n_covered else float("nan"),
            ece=metrics.expected_calibration_error(y_cov, s_cov, calibration_bins) if n_covered else float("nan"),
            auroc=metrics.auroc(y_cov, s_cov) if n_covered else float("nan"),
            auprc=metrics.auprc(y_cov, s_cov) if n_covered else float("nan"),
            precision_at_0_5=thr.precision if thr else float("nan"),
            recall_at_0_5=thr.recall if thr else float("nan"),
        )

    return HarnessReport(
        model_version=getattr(model, "model_version", "unknown"),
        n_users=int(len(users)),
        n_points_total=n_points_total,
        horizon=horizon,
        lookback=lookback,
        per_type=per_type,
        generated_at=datetime.now(UTC).isoformat(),
    )
