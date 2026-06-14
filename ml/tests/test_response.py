"""Anti-hallucination contract: {answer, sources, confidence}.

Asserts that hallucination-prone edge cases return a fallback (and never invoke
the LLM / fabricate), and that grounded answers always carry sources.
"""

from mindmap_ml.evidence.retrieve import Evidence, evidence_for
from mindmap_ml.models.base import Prediction
from mindmap_ml.narrative.response import (
    INSUFFICIENT_EVIDENCE,
    build_response,
)


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


def test_grounded_answer_has_sources_and_confidence() -> None:
    ev = evidence_for("sleep_deficit", "migraine")
    fake = FakeClient(
        "Your logs show a possible pattern: shorter sleep tended to come before migraine days. "
        "This is not a diagnosis — you might consider a steadier sleep schedule."
    )
    r = build_response(_pred(), ev, client=fake)
    assert r.sources, "a grounded answer must cite sources"
    assert r.confidence == 0.7
    assert fake.calls == 1
    d = r.to_dict()
    assert set(d) == {"answer", "sources", "confidence"}


def test_no_evidence_returns_fallback_without_calling_llm() -> None:
    ungrounded = Evidence(factor="unknown_factor", outcome="migraine", prior=None, passages=[])
    fake = FakeClient("FABRICATED — should never be returned")
    r = build_response(_pred(), ungrounded, client=fake)
    assert r.answer == INSUFFICIENT_EVIDENCE  # exact fallback string
    assert r.sources == []
    assert r.confidence == 0.0
    assert fake.calls == 0  # the LLM is never invoked → cannot fabricate


def test_abstained_returns_not_enough_data_without_llm() -> None:
    ev = evidence_for("sleep_deficit", "migraine")
    fake = FakeClient("should not be used")
    r = build_response(_pred(abstained=True), ev, client=fake)
    assert "enough" in r.answer.lower()
    assert r.sources == []
    assert fake.calls == 0


def test_diagnostic_model_output_is_replaced_not_returned() -> None:
    ev = evidence_for("sleep_deficit", "migraine")
    fake = FakeClient("You have depression and should stop taking your medication.")
    r = build_response(_pred(), ev, client=fake)
    assert "you have depression" not in r.answer.lower()
    assert "stop taking" not in r.answer.lower()
    assert r.confidence <= 0.3  # blocked text is distrusted
