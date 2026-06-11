from datetime import date

import pytest

from mindmap_ml.schema import MindMapEntry, RangeError


def test_valid_entry_and_missingness_preserved() -> None:
    e = MindMapEntry(user_id="u1", entry_date=date(2025, 1, 1), sleep_minutes=420, anxiety=4)
    assert e.sleep_minutes == 420
    assert e.depression is None  # missing stays None, not 0


def test_out_of_range_rejected() -> None:
    with pytest.raises(RangeError):
        MindMapEntry(user_id="u1", entry_date=date(2025, 1, 1), sleep_quality=9)  # 1..5
    with pytest.raises(RangeError):
        MindMapEntry(user_id="u1", entry_date=date(2025, 1, 1), mood_valence=5)  # -3..3
    with pytest.raises(RangeError):
        MindMapEntry(user_id="u1", entry_date=date(2025, 1, 1), anxiety=11)  # 0..10


def test_from_row_parses_iso_date_and_ignores_unknown() -> None:
    e = MindMapEntry.from_row(
        {"user_id": "u1", "entry_date": "2025-02-03", "anxiety": 2, "unknown_col": 99}
    )
    assert e.entry_date == date(2025, 2, 3)
    assert e.anxiety == 2


def test_to_row_roundtrip_date() -> None:
    e = MindMapEntry(user_id="u1", entry_date=date(2025, 1, 1), migraine=True)
    row = e.to_row()
    assert row["entry_date"] == "2025-01-01"
    assert row["migraine"] is True
