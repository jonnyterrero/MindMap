# MindMap ML — Labels & Instruments

## Operational labels (self-derived) — NOT diagnoses
Computed deterministically from the user's own *future* logs (`features/engineering.add_forward_labels`).
They are operational app-risk labels for evaluation, not clinical truth.

| label | definition (event in the next window) | grounding |
|---|---|---|
| `label_migraine` | migraine == true | self-report |
| `label_anxiety` | anxiety ≥ 7 | self-report + literature threshold |
| `label_mood` | mood_valence < 0 OR depression ≥ 6 | self-report |
| `label_pain_flare` | migraine_intensity ≥ 6 | self-report |

Each surfaced *pattern* built on these must map to a curated evidence prior or a
retrieved passage (`evidence/`) before it may be shown — no uncited claims.

## Label quality (`labels/instruments.label_quality`)
Each derived label is flagged `high` / `partial` / `low` by how many core fields
(`sleep_minutes, anxiety, depression, mood_valence`) were logged that day, so eval
can down-weight thin days.

## Validated screening instruments (`labels/instruments`) — the trusted anchors
The highest-ROI label upgrade for the clinician summary. **Screening severity, not a diagnosis.**

- **PHQ-9** — Kroenke, Spitzer & Williams (2001). 9 items 0–3, total 0–27.
  Bands: 0–4 minimal · 5–9 mild · 10–14 moderate · 15–19 moderately severe · 20–27 severe.
  **Item 9 (self-harm) ≥ 1 → `suicidality_flag` → crisis routing.**
- **GAD-7** — Spitzer et al. (2006). 7 items 0–3, total 0–21.
  Bands: 0–4 minimal · 5–9 mild · 10–14 moderate · 15–21 severe.

Recommended cadence: weekly or biweekly during the onboarding logging commitment.

## What NOT to do
- Do not invent diagnostic labels from self-report.
- Do not hand-curate one user's data into a "gold set" and treat it as truth.
- Do not train a served model on synthetic labels (synthetic is for tests/validation/power only).
