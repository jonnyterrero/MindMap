"""Typed dataclasses mirroring the MindMap Supabase tables, with the exact
range checks from supabase/schema.sql. Missingness is preserved (``None`` is
distinct from ``0``) and out-of-range values are rejected, never silently
clamped — per the safety contract.

The *modeled unit* downstream is a daily frame: one row per (user, entry_date)
with the core entry fields plus a few columns merged in from related tables
(``body_pain`` from body sensations; ``pressure``/``humidity``/``temp_max``/
``pressure_change`` from weather; adherence/completion rates). ``MindMapEntry``
validates a single core entry; the daily frame is assembled by the synthetic
generator (Phase 1) or ``serving.supabase_io`` (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from typing import Any

# Inclusive (min, max) for each numeric daily field — mirrors the
# mindmap_entries CHECK constraints in supabase/schema.sql exactly.
ENTRY_RANGES: dict[str, tuple[float, float]] = {
    "sleep_minutes": (0, 24 * 60),
    "sleep_quality": (1, 5),
    "hrv": (0, 400),
    "mood_valence": (-3, 3),
    "anxiety": (0, 10),
    "depression": (0, 10),
    "mania": (0, 10),
    "focus": (0, 10),
    "productivity": (0, 100),
    "therapy_minutes": (0, 24 * 60),
    "outside_minutes": (0, 24 * 60),
    "migraine_intensity": (0, 10),
}

# Core numeric signals on the entry itself.
ENTRY_NUMERIC_FIELDS: tuple[str, ...] = tuple(ENTRY_RANGES.keys())

# Extra numeric columns merged into the daily frame from related tables.
MERGED_NUMERIC_FIELDS: tuple[str, ...] = (
    "body_pain",  # max logged body-sensation intensity that day (0..10)
    "pressure",  # barometric pressure (hPa)
    "humidity",  # %
    "temp_max",  # °C
    "pressure_change",  # hPa over 24h; negative = drop
    "med_adherence_rate",  # 0..1
    "routine_completion_rate",  # 0..1
)

# Every numeric column the feature layer may consume.
DAILY_NUMERIC_FIELDS: tuple[str, ...] = ENTRY_NUMERIC_FIELDS + MERGED_NUMERIC_FIELDS


class RangeError(ValueError):
    """Raised when a field value falls outside its schema-defined range."""


def _check_range(name: str, value: Any) -> None:
    if value is None:
        return  # missingness preserved
    lo, hi = ENTRY_RANGES[name]
    if not (lo <= value <= hi):
        raise RangeError(f"{name}={value!r} outside allowed range [{lo}, {hi}]")


@dataclass
class MindMapEntry:
    """One row of ``public.mindmap_entries`` (the core daily check-in)."""

    user_id: str
    entry_date: date
    sleep_minutes: int | None = None
    sleep_quality: int | None = None  # 1..5
    bed_time: str | None = None  # "HH:MM[:SS]"
    wake_time: str | None = None
    hrv: int | None = None  # 0..400
    mood_valence: int | None = None  # -3..3
    anxiety: int | None = None  # 0..10
    depression: int | None = None  # 0..10
    mania: int | None = None  # 0..10
    focus: int | None = None  # 0..10
    productivity: int | None = None  # 0..100
    therapy_minutes: int | None = None
    outside_minutes: int | None = None
    migraine: bool = False
    migraine_intensity: int | None = None  # 0..10
    migraine_aura: bool | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.entry_date, date):
            raise TypeError("entry_date must be a datetime.date")
        for name in ENTRY_NUMERIC_FIELDS:
            _check_range(name, getattr(self, name))

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> MindMapEntry:
        """Build from a Supabase/dict row, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in row.items() if k in known}
        ed = data.get("entry_date")
        if isinstance(ed, str):
            data["entry_date"] = date.fromisoformat(ed)
        return cls(**data)

    def to_row(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v.isoformat() if isinstance(v, date) else v
        return out


@dataclass
class BodySensation:
    """One row of ``public.mindmap_body_sensations`` (entry-linked)."""

    entry_id: str
    body_part: str
    intensity: int  # 0..10
    notes: str | None = None

    def __post_init__(self) -> None:
        if not (0 <= self.intensity <= 10):
            raise RangeError(f"intensity={self.intensity} outside [0, 10]")


@dataclass
class WeatherDaily:
    """Per-day weather merged into the daily frame by ``entry_date``."""

    entry_date: date
    pressure: float | None = None
    humidity: float | None = None
    temp_max: float | None = None
    pressure_change: float | None = None  # hPa/24h, negative = drop
    pollen_level: str | None = None  # low|moderate|high|very_high


@dataclass
class JournalEntry:
    """One row of ``public.mindmap_journal_entries``. Raw ``content`` is private
    and must never be persisted into ML tables or logs; only privacy-safe,
    aggregated features may leave this object (see privacy rules)."""

    user_id: str
    entry_date: date
    content: str
    title: str | None = None
    mood_tags: list[str] | None = None
    is_private: bool = True
