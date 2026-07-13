"""Obsidian adapter — markdown structure becomes metadata, never fake prose.

The cleaned text is the pipeline's raw_text, so the Stage-1 offset guarantee
must hold on it, and no content words may be lost in stripping.
"""

import re

from mindmap_ml.graph.ingest import digest, normalize
from mindmap_ml.graph.obsidian import (
    load_vault,
    parse_frontmatter,
    parse_note,
    run_pipeline_on_note,
)

NOTE = """---
tags: [health, sleep]
mood: "4"
aliases:
  - Daily Log
  - Journal
---

# Tuesday

I slept badly and felt anxious all day. #anxiety

- [x] took meds
- skipped [[Morning Routine|my routine]] again

> [!note] reflection
> Work stress is **really** building up.

```python
ignore = "this is code, not journal"
```

See [[Sleep Hygiene#Basics]] and [my doc](https://example.com/doc).
%% private comment %%

---
"""


def test_frontmatter_parsed_and_removed():
    fm, body = parse_frontmatter(NOTE)
    assert fm["tags"] == ["health", "sleep"]
    assert fm["mood"] == "4"
    assert fm["aliases"] == ["Daily Log", "Journal"]
    assert "aliases" not in body and "Tuesday" in body


def test_note_structure_extracted():
    note = parse_note(NOTE, title="2026-07-12")
    assert note.title == "2026-07-12"
    # frontmatter + inline tags, deduped, no '#'
    assert note.tags == ["health", "sleep", "anxiety"]
    # wikilink targets collected; heading anchor and alias resolved to target
    assert note.wikilinks == ["Morning Routine", "Sleep Hygiene"]


def test_prose_is_clean():
    text = parse_note(NOTE).text
    assert "I slept badly and felt anxious all day." in text
    assert "took meds" in text
    assert "skipped my routine again" in text  # alias text kept, brackets gone
    assert "Work stress is really building up." in text  # quote + bold stripped
    assert "See Sleep Hygiene and my doc." in text  # links -> display text
    for junk in ("#anxiety", "[[", "```", "ignore =", "%%", "- [x]", "[!note]", ">"):
        assert junk not in text
    # heading text survives as a line; hrule and code do not
    assert "Tuesday" in text.split("\n")[0]


def test_no_content_words_lost():
    note = parse_note(NOTE)
    prose_words = {"slept", "badly", "anxious", "meds", "skipped", "routine", "stress", "building"}
    got = set(re.findall(r"[a-z]+", note.text.lower()))
    assert prose_words <= got


def test_offset_integrity_holds_on_cleaned_text():
    note = parse_note(NOTE)
    doc, spans = digest(note.text, user_id="u1", source_type="notes")
    assert spans
    for sp in spans:
        assert doc.norm_text[sp.norm_start : sp.norm_end] == sp.text
        assert normalize(doc.raw_text[sp.raw_start : sp.raw_end])[0] == sp.text


def test_end_to_end_pipeline_on_note():
    note, artifact = run_pipeline_on_note(NOTE, user_id="u1", title="daily")
    assert note.text
    assert artifact.user_id == "u1"
    # fail-closed shape: every surviving claim carries verifier confidence
    for n in artifact.nodes:
        assert n.status in ("verified", "downgraded")
        assert n.confidence is not None


def test_empty_note_abstains_not_fabricates():
    note, artifact = run_pipeline_on_note("---\ntags: [x]\n---\n", user_id="u1")
    assert note.text == ""
    assert artifact.nodes == [] and artifact.edges == []


def test_tag_vs_heading_disambiguation():
    note = parse_note("# Heading\nworked on #focus/deep today\n#2026 stays prose")
    assert "focus/deep" in note.tags
    assert "2026" not in note.tags  # numeric-start is not a tag
    assert "worked on today" in note.text
    assert "Heading" in note.text


def test_load_vault_skips_hidden_and_sorts(tmp_path):
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "app.md").write_text("config junk", encoding="utf-8")
    (tmp_path / "b.md").write_text("second note", encoding="utf-8")
    sub = tmp_path / "daily"
    sub.mkdir()
    (sub / "a.md").write_text("---\ntags: [t]\n---\nfirst note", encoding="utf-8")
    notes = load_vault(tmp_path)
    assert [n.path for n in notes] == ["b.md", "daily/a.md"]
    assert notes[1].title == "a" and notes[1].tags == ["t"]
