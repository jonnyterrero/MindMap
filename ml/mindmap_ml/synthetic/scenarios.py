"""Synthetic personas with KNOWN ground-truth effects.

Pre-launch there is no real data, so ground truth is the *only* way to validate a
model. Each persona embeds a recoverable-but-noisy generative rule (documented in
:data:`PERSONA_EFFECTS`). Effects are deliberately **lagged** — they depend on the
*previous* logged entry — so they line up with next-entry labels and reward a
model that uses temporal features over one that just reads today's score.

Every function takes ``(rng, n)`` and returns a dict of equal-length arrays, one
per daily-frame column. Determinism comes from the caller's seeded RNG.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Columns each persona must fill (NaN = field left blank that day).
COLUMNS: tuple[str, ...] = (
    "sleep_minutes",
    "sleep_quality",
    "hrv",
    "mood_valence",
    "anxiety",
    "depression",
    "mania",
    "focus",
    "productivity",
    "therapy_minutes",
    "outside_minutes",
    "migraine",
    "migraine_intensity",
    "migraine_aura",
    "body_pain",
    "pressure",
    "humidity",
    "temp_max",
    "pressure_change",
    "pollen_level",
    "med_adherence_rate",
    "routine_completion_rate",
    "notes",
)

PERSONA_EFFECTS: dict[str, str] = {
    "stable_mood": "No structured effect; low symptoms + noise. Models should mostly abstain or predict low.",
    "sleep_sensitive_migraineur": "migraine[t] driven by sleep[t-1] < 6h AND pressure_change[t-1] < -8 hPa.",
    "anxiety_after_poor_sleep": "anxiety[t] elevated when sleep[t-1] < 6h.",
    "depressive_worsening": "depression / low mood drift upward over the series.",
    "mania_activation": "clustered episodes: mania spikes with reduced sleep + elevated mood.",
    "sparse_logger": "stable dynamics but few logged days (cold-start / abstention territory).",
    "noisy_logger": "stable means, high variance + extra missingness (low signal).",
    "crisis_journal": "stable, but one journal note contains crisis language (exercises detection).",
}


def _clip(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(a, lo, hi)


def _pressure_series(rng: np.random.Generator, n: int, vol: float) -> tuple[np.ndarray, np.ndarray]:
    steps = rng.normal(0, vol, n)
    pressure = _clip(1013 + np.cumsum(steps), 980, 1040)
    change = np.empty(n)
    change[0] = 0.0
    change[1:] = np.diff(pressure)
    return pressure, change


def _base(rng: np.random.Generator, n: int, pressure_vol: float = 2.5) -> dict[str, np.ndarray]:
    pressure, pchange = _pressure_series(rng, n, pressure_vol)
    therapy = np.where(rng.random(n) < 0.1, 50.0, 0.0)
    migraine = rng.random(n) < 0.04
    intensity = np.where(migraine, _clip(np.round(rng.normal(5, 2, n)), 1, 10), np.nan)
    aura_draw = rng.random(n) < 0.3
    aura = np.array([bool(aura_draw[i]) if migraine[i] else None for i in range(n)], dtype=object)
    body = np.where(rng.random(n) < 0.25, _clip(np.round(rng.normal(3, 2, n)), 0, 10), np.nan)
    pollen = rng.choice(
        ["low", "moderate", "high", "very_high"], size=n, p=[0.5, 0.3, 0.15, 0.05]
    )
    return {
        "sleep_minutes": _clip(np.round(rng.normal(455, 35, n)), 240, 600),
        "sleep_quality": _clip(np.round(rng.normal(4, 0.7, n)), 1, 5),
        "hrv": _clip(np.round(rng.normal(62, 12, n)), 20, 130),
        "mood_valence": _clip(np.round(rng.normal(1, 1, n)), -3, 3),
        "anxiety": _clip(np.round(rng.normal(2.2, 1.3, n)), 0, 10),
        "depression": _clip(np.round(rng.normal(2.0, 1.3, n)), 0, 10),
        "mania": _clip(np.round(rng.normal(1.0, 1.0, n)), 0, 10),
        "focus": _clip(np.round(rng.normal(6.5, 1.3, n)), 0, 10),
        "productivity": _clip(np.round(rng.normal(62, 14, n)), 0, 100),
        "therapy_minutes": therapy,
        "outside_minutes": _clip(np.round(rng.normal(60, 30, n)), 0, 300),
        "migraine": migraine,
        "migraine_intensity": intensity,
        "migraine_aura": aura,
        "body_pain": body,
        "pressure": np.round(pressure, 1),
        "humidity": _clip(np.round(rng.normal(55, 15, n)), 10, 100),
        "temp_max": _clip(np.round(rng.normal(20, 7, n)), -5, 40),
        "pressure_change": np.round(pchange, 1),
        "pollen_level": pollen,
        "med_adherence_rate": _clip(np.round(rng.normal(0.9, 0.08, n), 2), 0, 1),
        "routine_completion_rate": _clip(np.round(rng.normal(0.8, 0.12, n), 2), 0, 1),
        "notes": np.array([None] * n, dtype=object),
    }


def _inject_missing(rng: np.random.Generator, cols: dict[str, np.ndarray], rate: float) -> None:
    """Blank out some optional fields (preserving missingness as NaN)."""
    for key in ("hrv", "sleep_quality", "focus", "productivity", "outside_minutes"):
        mask = rng.random(len(cols[key])) < rate
        cols[key] = np.where(mask, np.nan, cols[key])


def _recompute_migraine_details(rng: np.random.Generator, cols: dict[str, np.ndarray]) -> None:
    mig = cols["migraine"].astype(bool)
    n = len(mig)
    cols["migraine_intensity"] = np.where(mig, _clip(np.round(rng.normal(6, 2, n)), 1, 10), np.nan)
    aura_draw = rng.random(n) < 0.35
    cols["migraine_aura"] = np.array(
        [bool(aura_draw[i]) if mig[i] else None for i in range(n)], dtype=object
    )


# --------------------------------------------------------------------------- #
# Personas
# --------------------------------------------------------------------------- #
def stable_mood(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    cols = _base(rng, n)
    _inject_missing(rng, cols, 0.05)
    return cols


def sleep_sensitive_migraineur(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    cols = _base(rng, n, pressure_vol=5.0)
    # frequent short-sleep nights
    short = rng.random(n) < 0.3
    cols["sleep_minutes"] = np.where(short, _clip(np.round(rng.normal(330, 35, n)), 180, 600), cols["sleep_minutes"])
    sleep = cols["sleep_minutes"]
    pchange = cols["pressure_change"]
    mig = np.zeros(n, dtype=bool)
    for t in range(1, n):
        trigger = sleep[t - 1] < 360 and pchange[t - 1] < -8
        mig[t] = rng.random() < (0.8 if trigger else 0.04)
    cols["migraine"] = mig
    _recompute_migraine_details(rng, cols)
    _inject_missing(rng, cols, 0.05)
    return cols


def anxiety_after_poor_sleep(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    cols = _base(rng, n)
    short = rng.random(n) < 0.35
    cols["sleep_minutes"] = np.where(short, _clip(np.round(rng.normal(320, 40, n)), 180, 600), cols["sleep_minutes"])
    sleep = cols["sleep_minutes"]
    anx = cols["anxiety"].copy()
    for t in range(1, n):
        if sleep[t - 1] < 360:
            anx[t] = np.clip(round(2 + 6 + rng.normal(0, 1)), 0, 10)
    cols["anxiety"] = anx
    _inject_missing(rng, cols, 0.05)
    return cols


def depressive_worsening(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    cols = _base(rng, n)
    ramp = np.arange(n) / max(n - 1, 1)
    cols["depression"] = _clip(np.round(2 + 6 * ramp + rng.normal(0, 1, n)), 0, 10)
    cols["mood_valence"] = _clip(np.round(2 - 4 * ramp + rng.normal(0, 0.8, n)), -3, 3)
    cols["focus"] = _clip(np.round(7 - 3 * ramp + rng.normal(0, 1, n)), 0, 10)
    _inject_missing(rng, cols, 0.05)
    return cols


def mania_activation(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    cols = _base(rng, n)
    mania = cols["mania"].copy()
    sleep = cols["sleep_minutes"].copy()
    mood = cols["mood_valence"].copy()
    n_ep = max(1, n // 40)
    for _ in range(n_ep):
        start = int(rng.integers(0, max(1, n - 5)))
        length = int(rng.integers(3, 6))
        for t in range(start, min(n, start + length)):
            mania[t] = np.clip(round(rng.normal(8, 1)), 0, 10)
            sleep[t] = np.clip(round(rng.normal(300, 40)), 180, 600)
            mood[t] = np.clip(round(rng.normal(2.5, 0.6)), -3, 3)
    cols["mania"], cols["sleep_minutes"], cols["mood_valence"] = mania, sleep, mood
    _inject_missing(rng, cols, 0.05)
    return cols


def sparse_logger(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    # dynamics like stable; sparseness comes from the generator's log density
    cols = _base(rng, n)
    _inject_missing(rng, cols, 0.1)
    return cols


def noisy_logger(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    cols = _base(rng, n)
    for key, lo, hi in (("anxiety", 0, 10), ("depression", 0, 10), ("mood_valence", -3, 3), ("focus", 0, 10)):
        cols[key] = _clip(np.round(cols[key] + rng.normal(0, 2, n)), lo, hi)
    _inject_missing(rng, cols, 0.18)
    return cols


def crisis_journal(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    cols = _base(rng, n)
    k = int(rng.integers(0, n))
    notes = cols["notes"]
    notes[k] = "Some days I feel hopeless and like I can't cope anymore."
    cols["notes"] = notes
    _inject_missing(rng, cols, 0.05)
    return cols


PersonaFn = Callable[[np.random.Generator, int], dict[str, np.ndarray]]


# name -> (generator, n_users, log_density)
PERSONAS: dict[str, tuple[PersonaFn, int, float]] = {
    "stable_mood": (stable_mood, 4, 0.92),
    "sleep_sensitive_migraineur": (sleep_sensitive_migraineur, 4, 0.92),
    "anxiety_after_poor_sleep": (anxiety_after_poor_sleep, 4, 0.92),
    "depressive_worsening": (depressive_worsening, 3, 0.92),
    "mania_activation": (mania_activation, 3, 0.92),
    "sparse_logger": (sparse_logger, 3, 0.45),
    "noisy_logger": (noisy_logger, 3, 0.85),
    "crisis_journal": (crisis_journal, 2, 0.92),
}
