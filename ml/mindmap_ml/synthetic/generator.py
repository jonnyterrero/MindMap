"""Deterministic synthetic dataset assembly.

Walks each persona × user, picks which calendar days were *logged* (sparse
loggers log fewer), runs the persona's generative rule, and stitches the result
into one daily frame: one row per logged (user, entry_date). Marked clearly as
synthetic via the ``persona`` column; never to be mixed with real data.

Deterministic under ``seed`` (fixed RNG consumption order + fixed base date).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .scenarios import COLUMNS, PERSONAS

DEFAULT_BASE_DATE = date(2025, 1, 1)
SYNTHETIC_MARKER = True  # rows are fake; see persona column


def generate_dataset(
    seed: int = 0,
    n_days: int = 150,
    base_date: date = DEFAULT_BASE_DATE,
    min_entries: int = 5,
) -> pd.DataFrame:
    """Return a deterministic synthetic daily frame across all personas."""
    rng = np.random.default_rng(seed)
    chunks: list[pd.DataFrame] = []

    for persona_name, (fn, n_users, density) in PERSONAS.items():
        for u in range(n_users):
            logged = rng.random(n_days) < density
            offsets = np.flatnonzero(logged)
            if len(offsets) < min_entries:
                offsets = np.arange(min_entries)
            n = len(offsets)

            cols = fn(rng, n)
            data: dict[str, object] = {
                "user_id": [f"{persona_name}_{u:02d}"] * n,
                "entry_date": [base_date + timedelta(days=int(o)) for o in offsets],
                "persona": [persona_name] * n,
            }
            for c in COLUMNS:
                data[c] = cols[c]
            chunks.append(pd.DataFrame(data))

    df = pd.concat(chunks, ignore_index=True)
    # Stable, schema-faithful dtypes.
    df["migraine"] = df["migraine"].astype(bool)
    return df.sort_values(["user_id", "entry_date"]).reset_index(drop=True)
