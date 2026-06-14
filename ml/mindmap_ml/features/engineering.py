"""Pure feature transforms. No I/O, deterministic.

Operates on the **daily frame**: one row per *logged* (user, entry_date),
sorted ascending. Temporal features (lags, rolling, deltas) are computed over
each user's logged entries in order — the same row-based windowing the app's TS
engines use (``entries.slice(0, 7)`` = last 7 logged entries), not a calendar
reindex. Missingness flags capture fields a user left blank on a logged day.

Everything here is referentially transparent: same input frame → same output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .spec import DEFAULT_SPEC, FeatureSpec

USER_COL = "user_id"
DATE_COL = "entry_date"


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values([USER_COL, DATE_COL]).reset_index(drop=True)


def build_features(df: pd.DataFrame, spec: FeatureSpec = DEFAULT_SPEC) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered temporal columns appended.

    For each base column: ``{c}_lag{k}``, ``{c}_roll{mean,std}{w}``,
    ``{c}_delta{k}``, and ``{c}_missing``.
    """
    out = _sorted(df)
    grp = out.groupby(USER_COL, sort=False)

    # Build all engineered columns into one dict and concat once — avoids the
    # fragmented-frame PerformanceWarning and is much faster when called per point.
    new_cols: dict[str, pd.Series] = {}
    for col in spec.base_columns:
        if col not in out.columns:
            # Column may be absent for a given dataset; skip gracefully so the
            # spec stays a superset and tests stay robust to optional signals.
            continue
        s = out[col]

        for k in spec.lags:
            new_cols[f"{col}_lag{k}"] = grp[col].shift(k)

        for w in spec.rolling_windows:
            if "mean" in spec.rolling_stats:
                new_cols[f"{col}_rollmean{w}"] = grp[col].transform(
                    lambda x, w=w: x.rolling(w, min_periods=1).mean()
                )
            if "std" in spec.rolling_stats:
                new_cols[f"{col}_rollstd{w}"] = grp[col].transform(
                    lambda x, w=w: x.rolling(w, min_periods=2).std()
                )

        for k in spec.deltas:
            new_cols[f"{col}_delta{k}"] = s - grp[col].shift(k)

        if spec.add_missingness_flags:
            new_cols[f"{col}_missing"] = s.isna().astype("int8")

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return out


# --------------------------------------------------------------------------- #
# Forward labels (operational app-risk labels — NOT diagnoses).
# Aligned to the baseline's recurrence predicates so model & baseline are judged
# against a consistent target.
# --------------------------------------------------------------------------- #
def _label_conditions(df: pd.DataFrame) -> dict[str, pd.Series]:
    migraine = df.get("migraine", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    anxiety = df.get("anxiety")
    mood = df.get("mood_valence")
    depression = df.get("depression")
    intensity = df.get("migraine_intensity")

    def ge(series: pd.Series | None, thr: float) -> pd.Series:
        if series is None:
            return pd.Series(False, index=df.index)
        return (series >= thr).fillna(False)

    def lt(series: pd.Series | None, thr: float) -> pd.Series:
        if series is None:
            return pd.Series(False, index=df.index)
        return (series < thr).fillna(False)

    return {
        "label_migraine": migraine,
        "label_anxiety": ge(anxiety, 7),
        "label_mood": lt(mood, 0) | ge(depression, 6),
        "label_pain_flare": ge(intensity, 6),
    }


def add_forward_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Append binary forward labels: did the condition occur in the next
    ``horizon`` logged entries? The final ``horizon`` rows per user have no full
    forward window and get ``NaN`` (excluded from eval).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    out = _sorted(df)
    conds = _label_conditions(out)

    grp_pos = out.groupby(USER_COL, sort=False).cumcount()
    grp_size = out.groupby(USER_COL, sort=False)[DATE_COL].transform("size")
    tail = grp_pos >= (grp_size - horizon)

    for name, cond in conds.items():
        tmp = out[[USER_COL]].copy()
        tmp["_c"] = cond.astype(float).to_numpy()
        shifts = [
            tmp.groupby(USER_COL, sort=False)["_c"].shift(-k) for k in range(1, horizon + 1)
        ]
        any_future = pd.concat(shifts, axis=1).max(axis=1)  # skipna; all-NaN tail -> NaN
        lab = (any_future.fillna(0) >= 1).astype(float)
        lab[tail] = np.nan
        out[name] = lab.to_numpy()

    return out
