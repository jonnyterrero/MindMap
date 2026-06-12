"""Narrative guardrails — the last line before text reaches a user.

Wraps the Phase-1 output gate and adds the citation requirement: a
recommendation may only stand if it carries a citation. Anything that trips the
banned-phrase / missing-framing checks, or a recommendation with no citation, is
suppressed and replaced with a safe fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..safety.gate import SUPPRESSED_REPLACEMENT, check_output


@dataclass
class GuardResult:
    allowed: bool
    text: str
    violations: list[str]


def enforce(text: str, *, is_recommendation: bool, citations: list[str]) -> GuardResult:
    """Validate user-facing narrative text.

    - banned diagnostic/medical/causal phrasing → suppress
    - risk claim missing uncertainty framing → suppress
    - a recommendation with no citation → suppress (no uncited advice)
    """
    res = check_output(text, is_risk_claim=True)
    if not res.allowed:
        return GuardResult(False, res.safe_text or SUPPRESSED_REPLACEMENT, res.violations)
    if is_recommendation and not citations:
        return GuardResult(False, SUPPRESSED_REPLACEMENT, ["uncited_recommendation"])
    return GuardResult(True, text, [])
