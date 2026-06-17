"""Stage 1 (graph pipeline) — offset integrity is the load-bearing invariant.

If span offsets drift, every downstream provenance guarantee is void, so these
tests are deliberately aggressive about round-trip exactness and reversibility.
"""

import re

import pytest

from mindmap_ml.graph.ingest import canonicalize, digest, normalize, segment

TRICKY_INPUTS = [
    "I feel anxious today. Work was hard. I slept badly.",
    "# My day\n- woke up late\n- skipped breakfast\nfelt rushed",
    "I said “hello” — it’s fine…",  # smart quotes, em dash, ellipsis
    "line one\t\tword\n\n\nline two",  # tabs + blank lines
    "tired\nstressed\nhopeful",  # fragments, no punctuation
    "a    b   c",  # collapsed runs of spaces
    "café résumé",  # accented (NFC)
    "café",  # decomposed é -> canonicalizes to café
    "Trailing spaces here.   \n\n  Leading too.",
]


def _assert_offset_integrity(text: str):
    doc, spans = digest(text, user_id="u1")
    for sp in spans:
        # 1) norm round-trip is exact
        assert doc.norm_text[sp.norm_start : sp.norm_end] == sp.text
        # 2) raw bounds are valid
        assert 0 <= sp.raw_start < sp.raw_end <= len(doc.raw_text)
        # 3) reversibility: re-normalizing the raw region reproduces the span text
        assert normalize(doc.raw_text[sp.raw_start : sp.raw_end])[0] == sp.text
    # 4) strictly increasing, non-overlapping by raw and norm
    for a, b in zip(spans, spans[1:], strict=False):
        assert a.raw_start < b.raw_start
        assert a.norm_start < b.norm_start
        assert a.raw_end <= b.raw_start
    # 5) no content word is dropped (only formatting/whitespace may be)
    span_tokens = set()
    for sp in spans:
        span_tokens |= set(re.findall(r"[a-z0-9]+", sp.text.lower()))
    assert set(re.findall(r"[a-z0-9]+", doc.norm_text.lower())) == span_tokens
    return doc, spans


@pytest.mark.parametrize("text", TRICKY_INPUTS)
def test_offset_integrity(text: str) -> None:
    _assert_offset_integrity(text)


def test_raw_text_is_nfc_canonical() -> None:
    doc, spans = digest("café", user_id="u1")  # decomposed é
    assert doc.raw_text == "café"  # composed (NFC), length 4
    assert len(doc.raw_text) == 4
    assert spans[0].text == "café"
    # highlighting the original works directly off raw offsets
    sp = spans[0]
    assert doc.raw_text[sp.raw_start : sp.raw_end] == "café"


def test_three_sentences() -> None:
    doc, spans = digest("I feel anxious today. Work was hard. I slept badly.", user_id="u1")
    assert len(spans) == 3
    assert [s.text for s in spans] == ["I feel anxious today.", "Work was hard.", "I slept badly."]
    assert all(s.unit_type == "sentence" for s in spans)


def test_fragments_preserved_per_line() -> None:
    doc, spans = digest("tired\nstressed\nhopeful", user_id="u1")
    assert [s.text for s in spans] == ["tired", "stressed", "hopeful"]
    assert all(s.unit_type == "fragment" for s in spans)


def test_bullets_split_by_line() -> None:
    doc, spans = digest("# My day\n- woke up late\n- skipped breakfast\nfelt rushed", user_id="u1")
    assert [s.text for s in spans] == ["# My day", "- woke up late", "- skipped breakfast", "felt rushed"]


def test_typography_normalized_1to1() -> None:
    doc, spans = digest("I said “hello” — it’s fine.", user_id="u1")
    text = spans[0].text
    assert "“" not in text and "”" not in text and "’" not in text and "—" not in text
    assert '"hello"' in text and "it's" in text and "-" in text


def test_whitespace_collapsed_but_reversible() -> None:
    doc, spans = digest("a    b   c", user_id="u1")
    assert spans[0].text == "a b c"
    assert normalize(doc.raw_text[spans[0].raw_start : spans[0].raw_end])[0] == "a b c"


def test_determinism() -> None:
    text = "# My day\n- woke up late.\nI feel ok though."
    a_doc, a_spans = digest(text, user_id="u1", created_at="2025-01-01T00:00:00Z")
    b_doc, b_spans = digest(text, user_id="u1", created_at="2025-01-01T00:00:00Z")
    assert a_doc.to_dict() == b_doc.to_dict()
    assert [s.to_dict() for s in a_spans] == [s.to_dict() for s in b_spans]


def test_empty_and_whitespace_inputs() -> None:
    for blank in ["", "   ", "  \n\t \n "]:
        doc, spans = digest(blank, user_id="u1")
        assert spans == []
        assert segment(doc.norm_text) == []


def test_ids_stable_and_formatted() -> None:
    doc, spans = digest("hello world.", user_id="u1", doc_id=None)
    assert doc.doc_id.startswith("doc_") and len(doc.doc_id) == 16
    assert spans[0].span_id == f"sp_{doc.doc_id}_0"
    # doc_id is content-addressed (stable across runs / users)
    again, _ = digest("hello world.", user_id="someone_else")
    assert again.doc_id == doc.doc_id


def test_canonicalize_handles_none() -> None:
    assert canonicalize(None) == ""
    doc, spans = digest("", user_id="u1")
    assert doc.norm_text == "" and spans == []
