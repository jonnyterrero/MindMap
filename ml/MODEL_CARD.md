# MindMap ML — Model Card

## What it predicts

Per-user, next-day **operational risk trends** for four outcomes: `migraine`, `anxiety`, `mood`,
`pain_flare`. Outputs are calibrated probabilities (0–1) with a risk level, confidence/uncertainty,
contributing factors, and (when grounded) evidence citations.

## What it does NOT predict / do

It does not diagnose, label, or imply any medical/psychiatric condition; it gives no treatment or
medication advice; it makes no causal claims. Labels are **operational app-risk labels**, not
diagnoses (e.g. `label_anxiety` = an anxiety score ≥ 7 occurs in the next entry).

## Data

- **Used:** the user's own logged daily entries (sleep, mood, anxiety, depression, mania, focus,
  productivity, migraine + intensity, HRV), plus merged body-pain, weather, and adherence/routine
  signals — engineered into lags, rolling windows, deltas, and missingness flags.
- **Never used:** raw private journal text is not used as a model input or persisted to ML tables.
- **Pre-launch:** validated entirely on **synthetic data** with known ground-truth personas
  (`synthetic/scenarios.py`); the synthetic→real swap is a one-file change at the Supabase read.

## Models

- **Baseline (`v1_rule_extended`):** faithful Python port of the app's TS engines. The model-to-beat
  and the cold-start fallback.
- **ML (`v2_ml_assistive`):** per-type calibrated logistic regression (imputer → scaler → L2-logistic,
  sigmoid calibration) + lightweight per-user shrinkage toward the user's recent base rate.

## Evaluation (leave-user-out on synthetic, seed 0)

| type | AUROC base→ML | ECE base→ML |
|---|---|---|
| migraine | 0.49 → ~0.5 | 0.14 → **~0.00** |
| anxiety | 0.78 → **0.92** | 0.21 → 0.03 |
| mood | 0.59 → **0.75** | 0.16 → 0.03 |
| pain_flare | 0.48 → 0.57 | 0.26 → **0.00** |

Reproduce: `uv run python -m mindmap_ml.eval.compare`.

## Calibration & uncertainty

Probabilities are calibrated (sigmoid); the harness reports ECE/reliability and abstention coverage.
The ML model is dramatically better calibrated than the baseline across all types.

## Known limitations

- **Next-day migraine ranking is not reliably better than chance** (AUROC ~0.44–0.52 depending on
  split). The ML model wins on calibration, not ranking, for migraine. Treat migraine as
  rules-assisted / low-confidence and lean on abstention. This is the clearest candidate to keep in
  `rules_only` until more (real) data is available.
- Validated on synthetic data only; real-world performance is unknown until real data is collected.
- Per-user adaptation is a simple shrinkage, not a true hierarchical model.

## Human-in-the-loop

All outputs are framed as patterns to discuss with a qualified professional; elevated/high risk adds
clinician-review language. Crisis language routes to resources, bypassing normal output.

## How to disable

Run `rules_only` (pass no model to the batch job) to disable all ML outputs; the rule engine still
produces conservative, gated rows.
