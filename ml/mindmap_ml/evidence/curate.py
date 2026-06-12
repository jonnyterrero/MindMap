"""Curated evidence priors — a small, auditable structured table.

A recommendation may only be surfaced if it maps to one of these priors (or a
retrieved passage). Priors are loaded from a human-readable JSON file and
validated against a strict schema so the table can be reviewed like any other
clinical-content artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import ML_ROOT

SEED_PRIORS_PATH = ML_ROOT / "mindmap_ml" / "evidence" / "seed" / "priors.json"

DIRECTIONS = ("+", "-", "none")
STRENGTHS = ("weak", "moderate", "strong")
# Outcomes a prior may speak to — aligned with the model's prediction types plus
# the additional trackable outcomes the priors cover.
VALID_OUTCOMES = ("migraine", "anxiety", "mood", "pain_flare", "mania", "depression", "sleep")


class PriorValidationError(ValueError):
    pass


@dataclass(frozen=True)
class EvidencePrior:
    factor: str
    outcome: str
    direction: str  # + | - | none
    strength: str  # weak | moderate | strong
    citation: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.factor:
            raise PriorValidationError("prior.factor is required")
        if self.outcome not in VALID_OUTCOMES:
            raise PriorValidationError(f"prior.outcome {self.outcome!r} not in {VALID_OUTCOMES}")
        if self.direction not in DIRECTIONS:
            raise PriorValidationError(f"prior.direction {self.direction!r} not in {DIRECTIONS}")
        if self.strength not in STRENGTHS:
            raise PriorValidationError(f"prior.strength {self.strength!r} not in {STRENGTHS}")
        if not self.citation:
            raise PriorValidationError("prior.citation is required (no uncited priors)")


@lru_cache(maxsize=4)
def load_priors(path: Path = SEED_PRIORS_PATH) -> list[EvidencePrior]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("priors", data) if isinstance(data, dict) else data
    priors: list[EvidencePrior] = []
    for row in rows:
        priors.append(
            EvidencePrior(
                factor=row["factor"],
                outcome=row["outcome"],
                direction=row["direction"],
                strength=row["strength"],
                citation=row["citation"],
                note=row.get("note", ""),
            )
        )
    return priors


def find_prior(
    factor: str, outcome: str, priors: list[EvidencePrior] | None = None
) -> EvidencePrior | None:
    """Exact (factor, outcome) lookup — the grounding check for a recommendation."""
    priors = priors if priors is not None else load_priors()
    for p in priors:
        if p.factor == factor and p.outcome == outcome:
            return p
    return None
