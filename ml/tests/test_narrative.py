"""Phase 6: evidence-grounded narrative.

Golden tests (with an injected fake LLM — no key/network):
- diagnostic phrasing is blocked,
- abstained predictions yield a gentle "not enough data yet" message (no LLM call),
- every recommendation carries a citation,
- ungrounded predictions produce no recommendation.
"""

from mindmap_ml.evidence.retrieve import Evidence, evidence_for
from mindmap_ml.models.base import Prediction
from mindmap_ml.narrative.compose import compose_narrative
from mindmap_ml.narrative.guardrails import enforce


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return self.response


def _pred(abstained: bool = False) -> Prediction:
    if abstained:
        return Prediction("migraine", None, None, 0.3, 0.7, abstained=True, model_version="v2_ml_assistive")
    return Prediction("migraine", 0.45, "moderate", 0.7, 0.3, abstained=False, model_version="v2_ml_assistive")


def test_grounded_safe_narrative_carries_citation() -> None:
    ev = evidence_for("sleep_deficit", "migraine")
    safe = (
        "Your logs show a possible pattern: shorter sleep tended to come before migraine days. "
        "This is not a diagnosis — you might consider keeping a steadier sleep schedule."
    )
    fake = FakeClient(safe)
    n = compose_narrative(_pred(), ev, client=fake)
    assert fake.calls == 1
    assert not n.blocked and not n.abstained
    assert n.citations, "a grounded recommendation must carry a citation"


def test_diagnostic_output_is_blocked() -> None:
    ev = evidence_for("sleep_deficit", "migraine")
    fake = FakeClient("You have depression and you should stop taking your medication.")
    n = compose_narrative(_pred(), ev, client=fake)
    assert n.blocked is True
    assert "you have depression" not in n.text.lower()
    assert "stop taking" not in n.text.lower()


def test_abstained_is_gentle_and_skips_llm() -> None:
    ev = evidence_for("sleep_deficit", "migraine")
    fake = FakeClient("should never be used")
    n = compose_narrative(_pred(abstained=True), ev, client=fake)
    assert n.abstained is True
    assert fake.calls == 0  # no LLM call when abstaining
    assert "enough" in n.text.lower()  # "...not enough consistent data yet..."
    assert n.citations == []


def test_ungrounded_produces_no_recommendation() -> None:
    ev = Evidence(factor="totally_unknown", outcome="migraine", prior=None, passages=[])
    fake = FakeClient("ignored")
    n = compose_narrative(_pred(), ev, client=fake)
    assert fake.calls == 0  # no grounding -> no LLM, no recommendation
    assert n.citations == []
    assert "not a diagnosis" in n.text.lower()


def test_guardrails_block_uncited_recommendation() -> None:
    r = enforce(
        "You might consider a steadier routine. This is a pattern, not a diagnosis.",
        is_recommendation=True,
        citations=[],
    )
    assert r.allowed is False
    assert "uncited_recommendation" in r.violations
