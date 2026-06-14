# MindMap ML — Data Dictionary & Lineage

The modeled unit is a **daily frame**: one row per (user, `entry_date`). Source of
truth is `public.mindmap_entries` (Supabase); related tables are merged in by date.
Cheap to maintain now, essential for clinician credibility and acquisition due diligence.

## Core entry fields (from `mindmap_entries`)

| field | type | range | meaning | missing? |
|---|---|---|---|---|
| `sleep_minutes` | int | 0–1440 | total sleep | yes → NaN |
| `sleep_quality` | int | 1–5 | self-rated | yes |
| `hrv` | int | 0–400 | heart-rate variability (ms) | yes |
| `mood_valence` | int | −3..3 | mood | yes |
| `anxiety` | int | 0–10 | self-rated | yes |
| `depression` | int | 0–10 | self-rated | yes |
| `mania` | int | 0–10 | self-rated activation | yes |
| `focus` | int | 0–10 | self-rated | yes |
| `productivity` | int | 0–100 | self-rated | yes |
| `therapy_minutes` | int | 0–1440 | therapy time | yes |
| `outside_minutes` | int | 0–1440 | time outside | yes |
| `migraine` | bool | — | occurred that day | NOT NULL (false default) |
| `migraine_intensity` | int | 0–10 | if migraine | yes |
| `migraine_aura` | bool | — | if migraine | yes |
| `notes` | text | — | **private**; never an ML input/persisted feature | yes |

## Merged daily columns (from related tables)
`body_pain` (max body-sensation intensity that day, 0–10), `pressure` / `humidity` /
`temp_max` / `pressure_change` (weather), `med_adherence_rate`, `routine_completion_rate`.

## Conventions
- **Missingness is preserved** (`None`/NaN ≠ 0). `features/calendar.to_daily_calendar`
  reindexes to a gap-free daily calendar with a `logged` flag so gaps are visible.
- **Engineered features** (`features/engineering`, `features/spec`): per base signal —
  `{c}_lag{1,2,3}`, `{c}_roll{mean,std}{3,7,14,30}`, `{c}_delta{1,7}`, `{c}_missing`.
- **Forward labels** are operational, computed from the user's own future logs
  (`add_forward_labels`): `label_{migraine,anxiety,mood,pain_flare}` — see
  LABELS_AND_INSTRUMENTS.md. Operational, NOT diagnostic.

## Lineage
synthetic generator **or** `serving/supabase_io.read_entries` → `features/*` →
`insights/*` + `reports/clinician_summary` → (batch) `serving/score_batch` →
`mindmap_predictions`. Synthetic rows are marked by a `persona` column and never
mixed with real data.
