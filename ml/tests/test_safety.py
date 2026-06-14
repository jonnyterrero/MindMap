"""Safety tests: abstention contract, crisis detection, and the output gate.

Includes the mandated adversarial test — known unsafe strings MUST be blocked.
"""

import pytest

from mindmap_ml.safety.contract import (
    confidence_level,
    decide_abstention,
)
from mindmap_ml.safety.crisis import detect_crisis
from mindmap_ml.safety.gate import check_output


# --------------------------- abstention contract --------------------------- #
def test_abstain_on_insufficient_history() -> None:
    d = decide_abstention(history_days=3, confidence=0.9)
    assert d.abstain and d.reason == "insufficient_history"


def test_abstain_on_low_confidence() -> None:
    d = decide_abstention(history_days=30, confidence=0.2)
    assert d.abstain and d.reason == "low_confidence"


def test_abstain_on_missing_features_and_ood() -> None:
    assert decide_abstention(history_days=30, confidence=0.9, missing_key_features=True).reason == "missing_key_features"
    assert decide_abstention(history_days=30, confidence=0.9, out_of_distribution=True).reason == "out_of_distribution"


def test_no_abstain_when_healthy() -> None:
    d = decide_abstention(history_days=30, confidence=0.8)
    assert d.abstain is False and d.reason is None


def test_confidence_levels() -> None:
    assert confidence_level(0.9) == "high"
    assert confidence_level(0.5) == "medium"
    assert confidence_level(0.1) == "low"


# ------------------------------ crisis detect ------------------------------ #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("I want to die", "critical"),
        ("sometimes I want to hurt myself", "moderate"),
        ("I feel hopeless lately", "concern"),
        ("Today was a good day", None),
        ("", None),
        (None, None),
    ],
)
def test_detect_crisis(text, expected) -> None:
    assert detect_crisis(text) == expected


# ------------------------- output gate (adversarial) ----------------------- #
UNSAFE = [
    "You are bipolar.",
    "Stop taking your medication.",
    "Your sleep caused your depression.",
    "You should take Xanax.",
    "You definitely have ADHD.",
    "This is your diagnosis: major depression.",
    "You should increase your dose tonight.",
]


@pytest.mark.parametrize("text", UNSAFE)
def test_gate_blocks_unsafe_output(text) -> None:
    result = check_output(text, is_risk_claim=True)
    assert result.allowed is False
    assert result.safe_text  # a safe replacement is offered


def test_gate_allows_well_framed_pattern() -> None:
    safe = (
        "Your logs show a possible pattern: lower sleep tended to coincide with higher "
        "anxiety. This is not a diagnosis — consider discussing it with a professional."
    )
    assert check_output(safe, is_risk_claim=True).allowed is True


def test_gate_requires_uncertainty_framing_on_risk_claims() -> None:
    r = check_output("Anxiety is rising fast.", is_risk_claim=True)
    assert r.allowed is False and r.missing_framing is True


def test_gate_allows_clinician_attributed_treatment_plan() -> None:
    txt = "Bring this to your clinician to inform their treatment plan."
    assert check_output(txt, is_risk_claim=False).allowed is True
