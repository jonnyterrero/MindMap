import pytest

from mindmap_ml.labels.instruments import (
    InstrumentError,
    label_quality,
    score_gad7,
    score_phq9,
)


def test_phq9_scoring_and_bands() -> None:
    assert score_phq9([0] * 9).severity == "minimal"
    assert score_phq9([1] * 9).total == 9 and score_phq9([1] * 9).severity == "mild"
    assert score_phq9([2] * 9).total == 18 and score_phq9([2] * 9).severity == "moderately severe"
    assert score_phq9([3] * 9).total == 27 and score_phq9([3] * 9).severity == "severe"


def test_phq9_item9_safety_flag() -> None:
    safe = score_phq9([1, 1, 1, 1, 1, 1, 1, 1, 0])
    assert safe.suicidality_flag is False
    risk = score_phq9([0, 0, 0, 0, 0, 0, 0, 0, 2])
    assert risk.suicidality_flag is True and risk.item9 == 2


def test_gad7_scoring_and_bands() -> None:
    assert score_gad7([0] * 7).severity == "minimal"
    assert score_gad7([3] * 7).total == 21 and score_gad7([3] * 7).severity == "severe"


def test_instrument_validation() -> None:
    with pytest.raises(InstrumentError):
        score_phq9([0] * 8)  # wrong length
    with pytest.raises(InstrumentError):
        score_gad7([4, 0, 0, 0, 0, 0, 0])  # out of range
    with pytest.raises(InstrumentError):
        score_phq9([True] * 9)  # bool is not a valid item score


def test_not_a_diagnosis_disclaimer() -> None:
    assert "not a diagnosis" in score_phq9([0] * 9).disclaimer.lower()


def test_label_quality_flags() -> None:
    assert label_quality({"sleep_minutes": 400, "anxiety": 3, "depression": 2, "mood_valence": 1}) == "high"
    assert label_quality({"sleep_minutes": 400, "anxiety": 3, "depression": None, "mood_valence": None}) == "partial"
    assert label_quality({"sleep_minutes": None, "anxiety": None, "depression": None, "mood_valence": None}) == "low"
