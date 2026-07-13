"""Obsidian vault ingest adapter (pre-Stage-1).

Converts Obsidian-flavored markdown (YAML frontmatter, ``[[wikilinks]]``,
``#tags``, callouts, code fences) into clean prose plus structured metadata.
Stripping happens BEFORE Stage 1: the cleaned text becomes the pipeline's
``raw_text``, so the existing offset-integrity guarantee applies to it
unchanged (span offsets index the cleaned text, not the original .md bytes).

Pure stdlib, fully deterministic. Wikilink targets and tags are surfaced as
metadata — they are the user's own structure and useful downstream, but they
are never injected into the prose the generator reads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .generate import LLMClient
from .pipeline import run_pipeline
from .schema import MindmapArtifact
from .verify import Entailment

_FENCE_RE = re.compile(r"^(```|~~~).*?^\1\s*$\n?", re.DOTALL | re.MULTILINE)
_COMMENT_RE = re.compile(r"%%.*?%%", re.DOTALL)
_EMBED_RE = re.compile(r"!\[\[[^\]]*\]\]")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]*)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_TAG_RE = re.compile(r"(?<![\w#])#([A-Za-z][\w/-]*)")
_HEADING_RE = re.compile(r"^#{1,6}\s+")
_CHECKBOX_RE = re.compile(r"^(\s*)[-*+]\s+\[[ xX]\]\s+")
_LIST_RE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+")
_QUOTE_RE = re.compile(r"^\s*(?:>\s?)+")
_CALLOUT_RE = re.compile(r"^\[!\w+\][+-]?\s*")
_HRULE_RE = re.compile(r"^\s*([-*_])\s*(?:\1\s*){2,}$")
_EMPH_PAIR_RES = (
    re.compile(r"\*\*([^*]+)\*\*"),
    re.compile(r"__([^_]+)__"),
    re.compile(r"~~([^~]+)~~"),
    re.compile(r"==([^=]+)=="),
    re.compile(r"(?<!\w)\*([^*\n]+)\*(?!\w)"),
    re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)"),
)
_INLINE_CODE_RE = re.compile(r"`([^`\n]*)`")
_MULTI_SPACE_RE = re.compile(r" {2,}")


@dataclass
class ObsidianNote:
    """One parsed vault note: clean prose + the structure Obsidian encoded."""

    title: str
    text: str
    frontmatter: dict[str, str | list[str]] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "text": self.text,
            "frontmatter": self.frontmatter,
            "tags": self.tags,
            "wikilinks": self.wikilinks,
            "path": self.path,
        }


def _unquote(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def _parse_value(value: str) -> str | list[str]:
    v = value.strip()
    if v.startswith("[") and v.endswith("]"):
        items = [_unquote(x) for x in v[1:-1].split(",")]
        return [x for x in items if x]
    return _unquote(v)


def parse_frontmatter(markdown: str) -> tuple[dict[str, str | list[str]], str]:
    """Split leading YAML frontmatter from the body. Minimal parser (scalars,
    inline lists, block lists) — enough for vault metadata without a yaml dep."""
    lines = markdown.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, markdown
    close = next((i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None)
    if close is None:
        return {}, markdown

    fm: dict[str, str | list[str]] = {}
    key: str | None = None
    for line in lines[1:close]:
        item = re.match(r"^\s+-\s*(.*)$", line)
        if item and key is not None:
            existing = fm.get(key)
            block = existing if isinstance(existing, list) else []
            block.append(_unquote(item.group(1)))
            fm[key] = block
            continue
        kv = re.match(r"^([\w-]+)\s*:\s*(.*)$", line)
        if kv:
            key = kv.group(1)
            fm[key] = _parse_value(kv.group(2)) if kv.group(2).strip() else ""
    return fm, "\n".join(lines[close + 1 :])


def _frontmatter_tags(fm: dict[str, str | list[str]]) -> list[str]:
    raw = fm.get("tags", fm.get("tag", []))
    parts = raw if isinstance(raw, list) else re.split(r"[,\s]+", raw)
    return [p.lstrip("#") for p in parts if p.lstrip("#")]


def parse_note(markdown: str, *, title: str = "") -> ObsidianNote:
    """Obsidian markdown -> (clean prose, frontmatter, tags, wikilink targets)."""
    fm, body = parse_frontmatter(markdown)
    body = _COMMENT_RE.sub("", body)
    body = _FENCE_RE.sub("", body)
    body = _EMBED_RE.sub("", body)
    body = _IMAGE_RE.sub("", body)

    wikilinks: list[str] = []

    def _wikilink(m: re.Match[str]) -> str:
        target = m.group(1).strip()
        if target and target not in wikilinks:
            wikilinks.append(target)
        return (m.group(2) or target or "").strip()

    body = _WIKILINK_RE.sub(_wikilink, body)
    body = _MD_LINK_RE.sub(r"\1", body)

    tags = _frontmatter_tags(fm)

    def _tag(m: re.Match[str]) -> str:
        if m.group(1) not in tags:
            tags.append(m.group(1))
        return ""

    out_lines: list[str] = []
    for line in body.split("\n"):
        if _HRULE_RE.match(line):
            continue
        line = _QUOTE_RE.sub("", line)
        line = _CALLOUT_RE.sub("", line)
        line = _CHECKBOX_RE.sub(r"\1", line)
        line = _LIST_RE.sub(r"\1", line)
        line = _HEADING_RE.sub("", line)
        line = _TAG_RE.sub(_tag, line)
        for pair in _EMPH_PAIR_RES:
            line = pair.sub(r"\1", line)
        line = _INLINE_CODE_RE.sub(r"\1", line)
        line = _MULTI_SPACE_RE.sub(" ", line).strip()
        out_lines.append(line)

    text = "\n".join(out_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return ObsidianNote(title=title, text=text, frontmatter=fm, tags=tags, wikilinks=wikilinks)


def load_vault(root: str | Path) -> list[ObsidianNote]:
    """Read every .md note under ``root`` (sorted, deterministic), skipping
    hidden directories like ``.obsidian`` / ``.trash``."""
    rootp = Path(root)
    notes: list[ObsidianNote] = []
    for p in sorted(rootp.rglob("*.md")):
        rel = p.relative_to(rootp)
        if any(part.startswith(".") for part in rel.parts):
            continue
        note = parse_note(p.read_text(encoding="utf-8"), title=p.stem)
        note.path = str(rel)
        notes.append(note)
    return notes


def run_pipeline_on_note(
    markdown: str,
    *,
    user_id: str,
    title: str = "",
    source_type: str = "notes",
    extractor_client: LLMClient | None = None,
    entailment: Entailment | None = None,
    dedup: bool = True,
) -> tuple[ObsidianNote, MindmapArtifact]:
    """Parse one Obsidian note and run the full verified-mindmap pipeline on
    its prose. Empty notes (all structure, no prose) still return an artifact —
    the pipeline abstains rather than fabricating content."""
    note = parse_note(markdown, title=title)
    artifact = run_pipeline(
        note.text,
        user_id=user_id,
        source_type=source_type,
        extractor_client=extractor_client,
        entailment=entailment,
        dedup=dedup,
    )
    return note, artifact
