"""Calibrated forecasting models (Phase 3).

Per prediction type, a population model = imputer → scaler → L2-logistic, wrapped
in probability **calibration** (Platt/sigmoid). On top sits a lightweight
**per-user adaptation**: the calibrated population probability is shrunk toward
the user's own recent event rate, with weight that grows as the user logs more.

Every output carries calibrated uncertainty and passes through the **abstention
contract** (thin history / low confidence / unavailable model → abstain). The
model is self-contained: at predict time it rebuilds the as-of-day feature row
from the window using the same `features` spec used in training, so it plugs into
the same harness as the rule baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import MIN_CONFIDENCE_TO_SURFACE, MIN_HISTORY_DAYS, ML_MODEL_VERSION
from ..eval.baseline import _level_for, _num
from ..features.engineering import _label_conditions, add_forward_labels, build_features
from ..features.spec import DEFAULT_SPEC, FeatureSpec
from ..safety.contract import decide_abstention
from .base import ContributingFactor, Prediction, PredictionWindow

USER_COL = "user_id"
DATE_COL = "entry_date"

PREDICTION_TYPES = ("migraine", "anxiety", "mood", "pain_flare")
LABELS = {t: f"label_{t}" for t in PREDICTION_TYPES}

# minimum positives in a class before we trust a trained calibrated model
_MIN_POSITIVES = 12
# per-user shrinkage: weight on the user's own base rate saturates at this
_PERSONAL_WEIGHT_K = 20.0
_PERSONAL_WEIGHT_MAX = 0.5


def _condition_rate(entries: list[dict[str, Any]], ptype: str) -> float:
    """The user's recent empirical event rate for a type (personal prior)."""
    if not entries:
        return 0.0
    def hit(e: dict[str, Any]) -> bool:
        if ptype == "migraine":
            return e.get("migraine") is True
        if ptype == "anxiety":
            return (_num(e.get("anxiety")) or 0) >= 7
        if ptype == "mood":
            return (_num(e.get("mood_valence")) or 0) < 0 or (_num(e.get("depression")) or 0) >= 6
        if ptype == "pain_flare":
            return (_num(e.get("migraine_intensity")) or 0) >= 6
        return False
    return sum(1 for e in entries if hit(e)) / len(entries)


class ForecastModel:
    """Calibrated population logistic + per-user shrinkage, per prediction type."""

    def __init__(
        self,
        *,
        spec: FeatureSpec = DEFAULT_SPEC,
        apply_abstention: bool = True,
        min_history_days: int = MIN_HISTORY_DAYS,
        random_state: int = 0,
    ) -> None:
        self.model_version = ML_MODEL_VERSION
        self.spec = spec
        self.apply_abstention = apply_abstention
        self.min_history_days = min_history_days
        self.random_state = random_state
        self.feature_cols: list[str] = []
        self.models: dict[str, CalibratedClassifierCV | None] = {}

    # ------------------------------- training ------------------------------ #
    def fit(self, df: pd.DataFrame) -> ForecastModel:
        labeled = df if any(c in df.columns for c in LABELS.values()) else add_forward_labels(df)
        feats = build_features(labeled, self.spec)
        candidate_cols = [c for c in self.spec.output_columns() if c in feats.columns]
        self.feature_cols = candidate_cols

        for t, label in LABELS.items():
            if label not in feats.columns:
                self.models[t] = None
                continue
            sub = feats[candidate_cols + [label]].dropna(subset=[label])
            y = sub[label].to_numpy().astype(int)
            x = sub[candidate_cols].to_numpy(dtype=float)
            n_pos = int(y.sum())
            if n_pos < _MIN_POSITIVES or n_pos == len(y):
                self.models[t] = None  # too few/again-degenerate -> unavailable, will abstain
                continue
            base = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )
            # sigmoid calibration is robust on modest, imbalanced data
            cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            cal.fit(x, y)
            self.models[t] = cal
        return self

    # ------------------------------ inference ------------------------------ #
    def _feature_row(self, window: PredictionWindow) -> np.ndarray | None:
        if not window.entries:
            return None
        chrono = list(reversed(window.entries))  # window is newest-first
        frame = pd.DataFrame(chrono)
        if "user_id" not in frame.columns:
            frame["user_id"] = window.user_id
        if "entry_date" not in frame.columns:
            frame["entry_date"] = range(len(frame))
        feats = build_features(frame, self.spec)
        present = [c for c in self.feature_cols if c in feats.columns]
        row = feats.iloc[[-1]].reindex(columns=self.feature_cols)
        return row.to_numpy(dtype=float) if present else None

    def predict(self, window: PredictionWindow) -> list[Prediction]:
        x_row = self._feature_row(window)
        history = window.history_days
        preds: list[Prediction] = []

        for t in PREDICTION_TYPES:
            clf = self.models.get(t)
            model_available = clf is not None and x_row is not None

            p_pop = (
                float(clf.predict_proba(x_row)[0, 1])
                if (clf is not None and x_row is not None)
                else 0.0
            )
            p_user = _condition_rate(window.entries, t)
            alpha = min(history / (history + _PERSONAL_WEIGHT_K), _PERSONAL_WEIGHT_MAX) if history else 0.0
            risk = (1 - alpha) * p_pop + alpha * p_user

            confidence = self._confidence(history, p_pop if model_available else 0.5)
            decision = (
                decide_abstention(
                    history_days=history,
                    confidence=confidence,
                    model_available=model_available,
                    min_history_days=self.min_history_days,
                )
                if self.apply_abstention
                else type("D", (), {"abstain": False, "reason": None})()
            )

            if decision.abstain:
                preds.append(
                    Prediction(
                        prediction_type=t,
                        risk=None,
                        risk_level=None,
                        confidence=round(confidence, 3),
                        uncertainty=round(1 - confidence, 3),
                        contributing_factors=[],
                        abstained=True,
                        model_version=self.model_version,
                        abstain_reason=decision.reason,
                    )
                )
                continue

            factors = [
                ContributingFactor("ml_population", round(p_pop, 3), f"calibrated model p={p_pop:.2f}"),
                ContributingFactor("personal_base_rate", round(alpha * p_user, 3), f"recent rate {p_user:.2f} (w={alpha:.2f})"),
            ]
            preds.append(
                Prediction(
                    prediction_type=t,
                    risk=round(float(np.clip(risk, 0, 1)), 3),
                    risk_level=_level_for(float(np.clip(risk, 0, 1))),
                    confidence=round(confidence, 3),
                    uncertainty=round(1 - confidence, 3),
                    contributing_factors=factors,
                    abstained=False,
                    model_version=self.model_version,
                )
            )
        return preds

    @staticmethod
    def _confidence(history: int, p_pop: float) -> float:
        vol = 0.3 * min(history, 30) / 30
        margin = 0.2 * (2 * abs(p_pop - 0.5))
        return float(np.clip(0.4 + vol + margin, 0.0, 1.0))

    # --------------------------- batch inference --------------------------- #
    def score_frame(self, df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
        """Vectorized scoring of a whole daily frame (the Phase-5 batch path and
        the fast eval path). Returns one row per (user, entry_date) with, per
        type: ``risk_{t}``, ``abstained_{t}``, ``confidence_{t}``, ``reason_{t}``.

        Features at row t use trailing windows (≤``lookback``), matching the
        per-window ``predict`` path, so batch and single-point agree.
        """
        feats = build_features(df, self.spec).sort_values([USER_COL, DATE_COL]).reset_index(drop=True)
        grp = feats.groupby(USER_COL, sort=False)
        history = (grp.cumcount() + 1).clip(upper=lookback)
        conds = _label_conditions(feats)  # current-day event indicators per type

        out = feats[[USER_COL, DATE_COL]].copy()
        out["history"] = history.to_numpy()
        x = feats.reindex(columns=self.feature_cols).to_numpy(dtype=float)
        hist = history.to_numpy(dtype=float)

        for t in PREDICTION_TYPES:
            clf = self.models.get(t)
            if clf is None:
                out[f"risk_{t}"] = np.nan
                out[f"abstained_{t}"] = True
                out[f"confidence_{t}"] = 0.4
                out[f"reason_{t}"] = "model_unavailable"
                continue

            p_pop = clf.predict_proba(x)[:, 1]
            cond = conds[LABELS[t]].astype(float)
            p_user = (
                feats.assign(_c=cond.to_numpy())
                .groupby(USER_COL, sort=False)["_c"]
                .transform(lambda s: s.rolling(lookback, min_periods=1).mean())
                .to_numpy()
            )
            alpha = np.minimum(hist / (hist + _PERSONAL_WEIGHT_K), _PERSONAL_WEIGHT_MAX)
            risk = np.clip((1 - alpha) * p_pop + alpha * p_user, 0, 1)
            confidence = np.clip(0.4 + 0.3 * np.minimum(hist, 30) / 30 + 0.2 * (2 * np.abs(p_pop - 0.5)), 0, 1)

            abstain = (hist < self.min_history_days) | (confidence < MIN_CONFIDENCE_TO_SURFACE)
            reason = np.where(
                hist < self.min_history_days,
                "insufficient_history",
                np.where(confidence < MIN_CONFIDENCE_TO_SURFACE, "low_confidence", ""),
            )
            out[f"risk_{t}"] = np.where(abstain, np.nan, risk)
            out[f"abstained_{t}"] = abstain
            out[f"confidence_{t}"] = np.round(confidence, 3)
            out[f"reason_{t}"] = reason
        return out
