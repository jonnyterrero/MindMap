import numpy as np

from mindmap_ml.schema import ENTRY_RANGES
from mindmap_ml.synthetic.generator import generate_dataset
from mindmap_ml.synthetic.scenarios import PERSONAS


def test_deterministic_under_seed() -> None:
    a = generate_dataset(seed=0, n_days=60)
    b = generate_dataset(seed=0, n_days=60)
    assert a.equals(b)


def test_different_seed_differs() -> None:
    a = generate_dataset(seed=0, n_days=60)
    c = generate_dataset(seed=1, n_days=60)
    assert not a.equals(c)


def test_all_personas_present() -> None:
    df = generate_dataset(seed=0, n_days=60)
    assert set(df["persona"].unique()) == set(PERSONAS.keys())


def test_values_respect_schema_ranges() -> None:
    df = generate_dataset(seed=0, n_days=90)
    for col, (lo, hi) in ENTRY_RANGES.items():
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=float)
        present = vals[~np.isnan(vals)]
        assert present.min() >= lo, f"{col} below {lo}"
        assert present.max() <= hi, f"{col} above {hi}"


def test_crisis_persona_has_crisis_note() -> None:
    df = generate_dataset(seed=0, n_days=90)
    notes = df[df["persona"] == "crisis_journal"]["notes"].dropna()
    assert any("hopeless" in str(n) for n in notes)
