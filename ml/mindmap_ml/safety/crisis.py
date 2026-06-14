"""Crisis detection — faithful port of frontend/lib/crisis-detection.ts.

NOT a clinical screen. A conservative, high-recall keyword matcher that surfaces
crisis resources when concerning language appears in journals/notes. Errs toward
showing help. Highest tier wins. Kept byte-for-byte aligned with the TS tiers so
the Python batch layer and the app never disagree about what counts as a crisis.

Lives separately from the output gate: this scans *input* text (what the user
wrote); the gate validates *output* text (what we would show back).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

CrisisSeverity = Literal["concern", "moderate", "critical"]

# Highest tier first. Matched case-insensitively as substrings on a
# whitespace-collapsed string. Mirrors crisis-detection.ts TIERS exactly.
_TIERS: tuple[tuple[CrisisSeverity, tuple[str, ...]], ...] = (
    (
        "critical",
        (
            "suicide",
            "kill myself",
            "killing myself",
            "end my life",
            "end it all",
            "don't want to live",
            "do not want to live",
            "dont want to live",
            "want to die",
            "better off dead",
        ),
    ),
    (
        "moderate",
        (
            "self-harm",
            "self harm",
            "hurt myself",
            "harm myself",
            "want to disappear",
            "cut myself",
        ),
    ),
    (
        "concern",
        (
            "hopeless",
            "overwhelmed",
            "can't cope",
            "cant cope",
            "can not cope",
            "no point",
            "no reason to go on",
        ),
    ),
)

_WS = re.compile(r"\s+")


def detect_crisis(text: str | None) -> CrisisSeverity | None:
    if not text:
        return None
    normalized = _WS.sub(" ", text.lower())
    for severity, phrases in _TIERS:
        if any(p in normalized for p in phrases):
            return severity
    return None


@dataclass(frozen=True)
class CrisisResource:
    label: str
    detail: str
    href: str | None = None


# US resources, matching the app. Numbers are verified constants; do not localize
# without verified per-country numbers (see SAFETY_POLICY).
CRISIS_RESOURCES: tuple[CrisisResource, ...] = (
    CrisisResource("988 Suicide & Crisis Lifeline", "Call or text 988 (US, 24/7)", "tel:988"),
    CrisisResource("Crisis Text Line", "Text HOME to 741741", "sms:741741?&body=HOME"),
    CrisisResource("Emergency services", "Call 911 if you are in immediate danger", "tel:911"),
)


def crisis_header(severity: CrisisSeverity) -> dict[str, str]:
    if severity == "critical":
        return {
            "title": "Your safety matters — help is available now",
            "body": (
                "What you wrote sounds really hard. You don't have to face it alone. "
                "Please reach out to one of these now."
            ),
        }
    if severity == "moderate":
        return {
            "title": "You deserve support",
            "body": (
                "It sounds like you're going through something painful. "
                "Talking to someone can help."
            ),
        }
    return {
        "title": "You're not alone",
        "body": "Tough moments are real. If things feel heavy, support is here whenever you want it.",
    }
