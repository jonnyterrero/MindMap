"""Corpus ingestion: turn papers/notes into normalized, citable passages.

Re-runnable and pure given a directory. Supports ``.md``/``.txt`` (and ``.pdf``
if ``pypdf`` is installed; otherwise PDFs are skipped with a note). Each file may
declare a citation via an HTML comment on the first line:

    <!-- citation: Author et al. (Year), Title -->

Chunking is paragraph-level (blank-line separated), which keeps passages short
and individually citable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..config import ML_ROOT

SEED_CORPUS_DIR = ML_ROOT / "mindmap_ml" / "evidence" / "seed" / "corpus"

_CITATION_RE = re.compile(r"<!--\s*citation:\s*(.+?)\s*-->", re.IGNORECASE)
_MIN_CHARS = 40


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    source: str  # file stem
    citation: str
    chunk: int


def _citation_for(text: str, fallback: str) -> str:
    m = _CITATION_RE.search(text)
    return m.group(1).strip() if m else fallback


def _strip_meta(text: str) -> str:
    text = _CITATION_RE.sub("", text)
    # drop leading markdown headers (lines starting with #)
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    return "\n".join(lines)


def _chunks(text: str) -> list[str]:
    paras = re.split(r"\n\s*\n", text)
    out: list[str] = []
    for p in paras:
        cleaned = re.sub(r"\s+", " ", p).strip()
        if len(cleaned) >= _MIN_CHARS:
            out.append(cleaned)
    return out


def _read_pdf(path: Path) -> str | None:
    try:
        import pypdf
    except ImportError:
        return None
    reader = pypdf.PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def ingest_file(path: Path) -> list[Passage]:
    if path.suffix.lower() in (".md", ".txt"):
        raw = path.read_text(encoding="utf-8")
    elif path.suffix.lower() == ".pdf":
        raw = _read_pdf(path) or ""
        if not raw:
            return []  # pypdf missing or empty PDF — skip silently, re-runnable later
    else:
        return []

    citation = _citation_for(raw, fallback=path.stem)
    body = _strip_meta(raw)
    passages: list[Passage] = []
    for i, chunk in enumerate(_chunks(body)):
        passages.append(
            Passage(id=f"{path.stem}#{i}", text=chunk, source=path.stem, citation=citation, chunk=i)
        )
    return passages


def ingest_corpus(corpus_dir: Path = SEED_CORPUS_DIR) -> list[Passage]:
    """Ingest every supported file under ``corpus_dir`` (sorted, deterministic)."""
    if not corpus_dir.exists():
        return []
    passages: list[Passage] = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file():
            passages.extend(ingest_file(path))
    return passages
