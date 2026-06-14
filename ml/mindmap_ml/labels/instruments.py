"""Validated screening instruments (PHQ-9, GAD-7) + label-quality flags.

These are the single highest-ROI label upgrade for the clinician-summary goal:
standardized, research-validated, deterministic, and exactly what a clinician
trusts. Severity bands are the published *screening* categories — they are NOT a
diagnosis, and outputs must always say so.

Scoring references:
- PHQ-9: Kroenke, Spitzer & Williams (2001). 9 items, 0–3 each, total 0–27.
  Bands: 0–4 minimal, 5–9 mild, 10–14 moderate, 15–19 moderately severe, 20–27 severe.
  Item 9 screens self-harm/suicidality → any score ≥ 1 raises a safety flag.
- GAD-7: Spitzer et al. (2006). 7 items, 0–3 each, total 0–21.
  Bands: 0–4 minimal, 5–9 mild, 10–14 moderate, 15–21 severe.
"""

from __future__ import annotations

from dataclasses import dataclass

PHQ9_ITEMS = 9
GAD7_ITEMS = 7
_ITEM_MIN, _ITEM_MAX = 0, 3

_PHQ9_BANDS = ((4, "minimal"), (9, "mild"), (14, "moderate"), (19, "moderately severe"), (27, "severe"))
_GAD7_BANDS = ((4, "minimal"), (9, "mild"), (14, "moderate"), (21, "severe"))


class InstrumentError(ValueError):
    pass


def _validate(responses: list[int], n_items: int) -> None:
    if len(responses) != n_items:
        raise InstrumentError(f"expected {n_items} items, got {len(responses)}")
    for i, v in enumerate(responses):
        if not isinstance(v, int) or isinstance(v, bool) or not (_ITEM_MIN <= v <= _ITEM_MAX):
            raise InstrumentError(f"item {i} = {v!r} must be an int in [{_ITEM_MIN}, {_ITEM_MAX}]")


def _band(total: int, bands: tuple[tuple[int, str], ...]) -> str:
    for hi, name in bands:
        if total <= hi:
            return name
    return bands[-1][1]


@dataclass(frozen=True)
class Phq9Result:
    total: int
    severity: str
    item9: int  # self-harm/suicidality item
    suicidality_flag: bool  # item9 >= 1 → surface crisis resources
    disclaimer: str = "Screening severity, not a diagnosis."


@dataclass(frozen=True)
class Gad7Result:
    total: int
    severity: str
    disclaimer: str = "Screening severity, not a diagnosis."


def score_phq9(responses: list[int]) -> Phq9Result:
    _validate(responses, PHQ9_ITEMS)
    total = sum(responses)
    return Phq9Result(
        total=total,
        severity=_band(total, _PHQ9_BANDS),
        item9=responses[8],
        suicidality_flag=responses[8] >= 1,
    )


def score_gad7(responses: list[int]) -> Gad7Result:
    _validate(responses, GAD7_ITEMS)
    total = sum(responses)
    return Gad7Result(total=total, severity=_band(total, _GAD7_BANDS))


# --------------------------------------------------------------------------- #
# Label-quality flags — weight self-derived labels by how complete the day was.
# --------------------------------------------------------------------------- #
CORE_FIELDS = ("sleep_minutes", "anxiety", "depression", "mood_valence")


def _present(value: object) -> bool:
    if value is None:
        return False
    try:
        return value == value  # False for NaN
    except Exception:
        return True


def label_quality(row: dict[str, object], fields: tuple[str, ...] = CORE_FIELDS) -> str:
    """high / partial / low, from the share of core fields logged that day."""
    if not fields:
        return "low"
    frac = sum(1 for f in fields if _present(row.get(f))) / len(fields)
    if frac >= 0.75:
        return "high"
    if frac >= 0.4:
        return "partial"
    return "low"
