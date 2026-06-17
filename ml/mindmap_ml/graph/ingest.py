"""Stage 1 — Import & Digest (deterministic).

Turns raw user text into immutable, char-offset-addressable evidence spans while
preserving exact provenance back to the canonicalized raw text.

Pipeline: canonicalize (NFC) -> normalize (drop controls/zero-width, collapse
whitespace, 1:1 typography, keep newline as a boundary) while tracking a
norm->raw index map -> segment into sentence/fragment units -> emit TextSpan[].

Pure stdlib, fully deterministic. Markdown stripping and ML segmentation are
deliberately deferred (future) so the offset guarantee stays simple and provable.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import UTC, datetime

from .schema import GRAPH_PIPELINE_VERSION, RawDocument, TextSpan

# Horizontal whitespace (newline handled separately as a unit boundary).
_H_WS = frozenset(" \t\r\f\v")
_ZERO_WIDTH = frozenset("​‌‍﻿")
# Same-length typography normalizations (keep offsets 1:1).
_TYPO = {
    "‘": "'", "’": "'", "‛": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ",  # nbsp (also caught as whitespace, kept for clarity)
}
_SENT_END_RE = re.compile(r"[.!?][\"')\]]*$")

_EN_STOP = frozenset(
    ["the", "a", "an", "and", "i", "to", "of", "it", "is", "in", "that", "you", "my", "for", "was", "on", "so", "but", "with", "this", "me", "we", "he", "she", "they"]
)


def canonicalize(text: str | None) -> str:
    """NFC-normalize. The result is the offset source of truth (RawDocument.raw_text)."""
    return unicodedata.normalize("NFC", text or "")


def normalize(canonical: str) -> tuple[str, list[int]]:
    """Return (norm_text, norm_to_raw) where norm_to_raw[i] is the index in
    ``canonical`` that norm_text[i] originated from (strictly increasing)."""
    out: list[str] = []
    nmap: list[int] = []
    i, n = 0, len(canonical)
    while i < n:
        ch = canonical[i]
        if ch == "\n" or ch in _H_WS:
            run_start = i
            has_nl = ch == "\n"
            i += 1
            while i < n and (canonical[i] == "\n" or canonical[i] in _H_WS):
                has_nl = has_nl or canonical[i] == "\n"
                i += 1
            out.append("\n" if has_nl else " ")
            nmap.append(run_start)
            continue
        if ch in _ZERO_WIDTH or unicodedata.category(ch) == "Cc":
            i += 1  # drop control / zero-width
            continue
        out.append(_TYPO.get(ch, ch))
        nmap.append(i)
        i += 1
    return "".join(out), nmap


def _split_line(norm: str, s: int, e: int) -> list[tuple[int, int]]:
    """Split a single line [s,e) into sentence units, then trim spaces."""
    units: list[tuple[int, int]] = []
    seg_start = j = s
    while j < e:
        if norm[j] in ".!?":
            k = j + 1
            while k < e and norm[k] in ".!?\"')]":
                k += 1
            if k >= e or norm[k] == " ":  # boundary only before a space / line end
                units.append((seg_start, k))
                seg_start = j = k
                continue
        j += 1
    if seg_start < e:
        units.append((seg_start, e))

    trimmed: list[tuple[int, int]] = []
    for a, b in units:
        while a < b and norm[a] == " ":
            a += 1
        while b > a and norm[b - 1] == " ":
            b -= 1
        if b > a:
            trimmed.append((a, b))
    return trimmed


def segment(norm: str) -> list[tuple[int, int, str]]:
    """Segment normalized text into (start, end, unit_type) units. Newlines are
    hard boundaries (preserves fragmented / line-per-thought notes)."""
    units: list[tuple[int, int, str]] = []
    line_start = 0
    for i in range(len(norm) + 1):
        if i == len(norm) or norm[i] == "\n":
            for s, e in _split_line(norm, line_start, i):
                unit_type = "sentence" if _SENT_END_RE.search(norm[s:e]) else "fragment"
                units.append((s, e, unit_type))
            line_start = i + 1
    return units


def detect_language(norm: str) -> tuple[str, float]:
    """Lightweight English-stopword heuristic. Future: a real detector (fastText)."""
    toks = re.findall(r"[a-zA-Z']+", norm)
    if not toks:
        return ("unknown", 0.0)
    ratio = sum(1 for t in toks if t.lower() in _EN_STOP) / len(toks)
    if ratio >= 0.08:
        return ("en", round(min(1.0, ratio * 4), 3))
    return ("unknown", round(ratio, 3))


def digest(
    raw_text: str,
    *,
    user_id: str,
    source_type: str = "journal",
    created_at: str | None = None,
    doc_id: str | None = None,
) -> tuple[RawDocument, list[TextSpan]]:
    """Stage-1 entry point: raw text -> (RawDocument, TextSpan[])."""
    canonical = canonicalize(raw_text)
    norm, nmap = normalize(canonical)
    language, confidence = detect_language(norm)
    did = doc_id or "doc_" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]

    spans: list[TextSpan] = []
    for order, (s, e, unit_type) in enumerate(segment(norm)):
        spans.append(
            TextSpan(
                span_id=f"sp_{did}_{order}",
                doc_id=did,
                text=norm[s:e],
                norm_start=s,
                norm_end=e,
                raw_start=nmap[s],
                raw_end=nmap[e - 1] + 1,
                unit_type=unit_type,
                order=order,
            )
        )

    doc = RawDocument(
        doc_id=did,
        user_id=user_id,
        source_type=source_type,
        raw_text=canonical,
        norm_text=norm,
        language=language,
        language_confidence=confidence,
        created_at=created_at or datetime.now(UTC).isoformat(),
        pipeline_version=GRAPH_PIPELINE_VERSION,
    )
    return doc, spans
