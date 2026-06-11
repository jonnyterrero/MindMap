"""Declarative feature specification — the single source of truth.

``engineering.build_features`` reads this; nothing else defines what features
exist. Keeping it declarative means the feature set is auditable and the
synthetic→real swap can't silently change the model's inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

# Base signals we build temporal features from. Subset of DAILY_NUMERIC_FIELDS —
# the ones with enough day-to-day signal to be worth lagging/rolling.
BASE_SIGNALS: tuple[str, ...] = (
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
    "pressure_change",
    "humidity",
    "temp_max",
)


@dataclass(frozen=True)
class FeatureSpec:
    base_columns: tuple[str, ...] = BASE_SIGNALS
    lags: tuple[int, ...] = (1, 2, 3)
    rolling_windows: tuple[int, ...] = (3, 7, 14, 30)
    deltas: tuple[int, ...] = (1, 7)  # x_t - x_{t-k}
    add_missingness_flags: bool = True
    # Per-day rolling stats; mean captures level, std captures volatility.
    rolling_stats: tuple[str, ...] = ("mean", "std")

    def output_columns(self) -> list[str]:
        """The full, ordered list of engineered column names (excluding the
        passthrough base columns). Used by tests and by ML models to assert the
        feature contract hasn't drifted."""
        cols: list[str] = []
        for c in self.base_columns:
            for k in self.lags:
                cols.append(f"{c}_lag{k}")
            for w in self.rolling_windows:
                for stat in self.rolling_stats:
                    cols.append(f"{c}_roll{stat}{w}")
            for k in self.deltas:
                cols.append(f"{c}_delta{k}")
            if self.add_missingness_flags:
                cols.append(f"{c}_missing")
        return cols


DEFAULT_SPEC = FeatureSpec()
