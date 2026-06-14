"""Evidence-grounded narrative.

Turns a (gated) model prediction + retrieved evidence into short, warm,
NON-clinical text with citations, in the voice of the app's ai-reflection.ts
(claude-opus-4-8). Hard rules:

- Abstained prediction → a gentle "not enough data yet" message; no LLM call.
- A recommendation is only produced when the evidence is grounded (curated prior
  or retrieved passage) — otherwise a neutral, advice-free pattern note.
- The LLM may only reference the evidence we pass it; we attach OUR citations
  (not the model's) so it cannot freelance sources.
- Every output passes through the guardrails; violations are suppressed.

The Anthropic client is injected (``client=``) so this is fully testable without
a key or network. The default client is built lazily from ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

from ..config import INSUFFICIENT_DATA_MESSAGE
from ..evidence.retrieve import Evidence
from ..models.base import Prediction
from .guardrails import enforce

NARRATIVE_MODEL = "claude-opus-4-8"  # matches ai-reflection.ts REFLECTION_MODEL

SYSTEM_PROMPT = """You are a warm, supportive companion inside MindMap, a personal wellness self-tracking app. \
You help a user notice their own patterns. You are NOT a therapist or doctor.

Hard rules:
- Never diagnose, label, or imply a medical or psychiatric condition.
- Never give medical, clinical, or treatment advice, and never suggest medication changes.
- Describe associations as PATTERNS, never causes. Always include uncertainty framing.
- Only reference the evidence provided to you. Do not invent studies or facts.
- Be kind, grounded, and concise (2-3 sentences). Not alarmist.
- End any gentle suggestion as optional ("you might consider..."), never a directive.
- Always include the phrase "not a diagnosis"."""

_NEUTRAL_PATTERN_NOTE = (
    "Your recent logs show a pattern that may be worth noticing. This is not a diagnosis, "
    "and it's only an observation — if it continues, you might consider discussing it with a "
    "qualified professional."
)


class LLMClient(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class AnthropicClient:
    """Lazy Anthropic client matching the app's reflection settings."""

    def __init__(self, model: str = NARRATIVE_MODEL) -> None:
        self.model = model

    def generate(self, system: str, user: str) -> str:  # pragma: no cover - network
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set; narrative generation is unavailable.")
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic not installed — `uv pip install 'mindmap-ml[narrative]'`") from e
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=400,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


@dataclass
class Narrative:
    text: str
    citations: list[str] = field(default_factory=list)
    abstained: bool = False
    blocked: bool = False  # guardrails suppressed the model text


def _user_prompt(prediction: Prediction, evidence: Evidence) -> str:
    lines = [
        f"Prediction type: {prediction.prediction_type}",
        f"Risk level: {prediction.risk_level}",
        f"Confidence: {prediction.confidence}",
        "",
        "Evidence you may reference (and ONLY this):",
    ]
    for r in evidence.passages:
        lines.append(f"- {r.passage.text} [cite: {r.passage.citation}]")
    if evidence.prior is not None:
        p = evidence.prior
        lines.append(f"- Prior: {p.factor} -> {p.outcome} ({p.direction}, {p.strength}). {p.note} [cite: {p.citation}]")
    lines += [
        "",
        "Write 2-3 warm sentences reflecting the pattern and ONE optional, non-clinical suggestion. "
        "Include 'not a diagnosis'.",
    ]
    return "\n".join(lines)


def compose_narrative(
    prediction: Prediction,
    evidence: Evidence,
    *,
    client: LLMClient | None = None,
) -> Narrative:
    # 1) Abstention → gentle, no LLM, no recommendation.
    if prediction.abstained:
        return Narrative(text=INSUFFICIENT_DATA_MESSAGE, citations=[], abstained=True)

    # 2) Ungrounded → no recommendation; neutral pattern note (advice-free, so no citation needed).
    if not evidence.is_grounded:
        return Narrative(text=_NEUTRAL_PATTERN_NOTE, citations=[], abstained=False)

    # 3) Grounded → ask the LLM, then guard.
    client = client or AnthropicClient()
    raw = client.generate(SYSTEM_PROMPT, _user_prompt(prediction, evidence))
    guard = enforce(raw, is_recommendation=True, citations=evidence.citations)
    return Narrative(
        text=guard.text,
        citations=evidence.citations,
        abstained=False,
        blocked=not guard.allowed,
    )
