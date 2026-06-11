"""Output-validation gate.

The last guard before any string reaches a user. It checks for banned
diagnostic/medical/causal phrasing and, for risk claims, for the presence of
required uncertainty framing. It can **suppress** (block) or **flag for
downgrade**. Phase 6 wires the LLM narrative through this; Phase 5 runs every
persisted insight summary through it.

Detection is conservative and high-recall: false positives (a blocked benign
phrase) are acceptable; false negatives (an unsafe phrase slipping through) are
not. Banned phrases are ported from the build prompt's SAFETY_POLICY (§9).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Phrases that must NEVER appear in user-facing output. Lowercased substring
# match on whitespace-collapsed text. Diagnosis, medication directives, causal
# overclaims, false certainty, treatment directives.
BANNED_SUBSTRINGS: tuple[str, ...] = (
    # diagnosis / labeling
    "you have depression",
    "you have anxiety",
    "you have bipolar",
    "you are bipolar",
    "you have adhd",
    "you definitely have",
    "this means you have",
    "you are entering mania",
    "you're entering mania",
    # NB: bare "diagnosis"/"diagnosed" are handled by _DIAGNOSIS_RE below so the
    # recommended safe framing "this is not a diagnosis" is NOT blocked.
    # medication directives
    "stop taking",
    "start taking",
    "increase your dose",
    "decrease your dose",
    "increase your medication",
    "decrease your medication",
    "this medication is right for you",
    "you should take medication",
    "you should stop therapy",
    # causal overclaims / false certainty
    "caused your",
    "this caused",
    "guaranteed",
    "will definitely",
    "cure",
    # treatment directives (clinician-created plans are described, not prescribed)
    "treatment plan",
)

# A risk/insight claim must contain at least one of these framings so it reads as
# a pattern, not a verdict.
REQUIRED_FRAMING_ANY: tuple[str, ...] = (
    "not a diagnosis",
    "pattern",
    "may ",
    "might",
    "tends to",
    "tended to",
    "could ",
    "consider",
    "discuss",
    "possible",
    "association",
)

_WS = re.compile(r"\s+")
# "treatment plan" is allowed only when explicitly attributed to a clinician.
_CLINICIAN_PLAN = re.compile(
    r"(clinician|doctor|therapist|provider|professional)[^.]*treatment plan"
    r"|treatment plan[^.]*(clinician|doctor|therapist|provider|professional)"
)
# Diagnostic claims are banned EXCEPT the negated safe framing "not a diagnosis".
_DIAGNOSIS_RE = re.compile(r"(?<!not a )diagnos(?:is|ed|e|ing)")


@dataclass
class GateResult:
    allowed: bool
    violations: list[str] = field(default_factory=list)
    missing_framing: bool = False
    # Safe replacement when the original is suppressed. Callers should surface
    # this instead of the offending text.
    safe_text: str | None = None

    @property
    def text(self) -> str | None:
        return self.safe_text


SUPPRESSED_REPLACEMENT = (
    "We spotted a pattern worth noticing, but we're holding back the wording here "
    "to stay safe and non-clinical. This is not a diagnosis — consider discussing "
    "ongoing patterns with a qualified professional."
)


def _banned_hits(text: str) -> list[str]:
    norm = _WS.sub(" ", text.lower())
    hits: list[str] = []
    for phrase in BANNED_SUBSTRINGS:
        if phrase not in norm:
            continue
        # Allow clinician-attributed "treatment plan".
        if phrase == "treatment plan" and _CLINICIAN_PLAN.search(norm):
            continue
        hits.append(phrase)
    if _DIAGNOSIS_RE.search(norm):
        hits.append("diagnosis_claim")
    return hits


def has_required_framing(text: str) -> bool:
    norm = _WS.sub(" ", text.lower())
    return any(f in norm for f in REQUIRED_FRAMING_ANY)


def check_output(text: str, *, is_risk_claim: bool = True) -> GateResult:
    """Validate one user-facing string.

    A banned phrase → suppressed outright. A risk claim missing uncertainty
    framing → blocked and replaced with the safe fallback (callers may instead
    choose to re-render with framing added).
    """
    violations = _banned_hits(text)
    missing_framing = is_risk_claim and not has_required_framing(text)

    if violations or missing_framing:
        return GateResult(
            allowed=False,
            violations=violations,
            missing_framing=missing_framing,
            safe_text=SUPPRESSED_REPLACEMENT,
        )
    return GateResult(allowed=True, safe_text=text)
