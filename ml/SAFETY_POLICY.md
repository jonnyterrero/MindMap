# MindMap ML — Safety Policy

This system is **decision support for self-tracking**, not a clinician. It must never
diagnose, imply a condition, or give treatment/medication advice.

## The non-negotiable contract

1. **Abstention first.** Insufficient history (< 7 days), low confidence, missing key
   features, out-of-distribution input, unavailable/uncalibrated model → **abstain** with
   "not enough data yet". Implemented in `safety/contract.py`; every model routes through it.
2. **Calibrated uncertainty.** Every prediction ships `confidence`/`uncertainty`. The eval
   harness measures calibration (ECE/reliability), not just accuracy.
3. **No diagnosis / no clinical claims / no treatment or medication advice.** Suggestions are
   gentle, optional, non-clinical.
4. **Evidence-grounded recommendations only.** A recommendation may surface only if it maps to a
   curated prior (`evidence/seed/priors.json`) or a retrieved passage. No uncited advice.
5. **Output gate.** `safety/gate.py` checks every user-facing string for banned phrasing and for
   required uncertainty framing; it can suppress or downgrade. `narrative/guardrails.py` adds the
   citation requirement.
6. **Provenance.** Every persisted row carries `model_version` and `source` (rules|ml).

## Banned phrasing (suppressed by the gate)

Diagnosis/labeling ("you have …", "you are bipolar", "you definitely have …", any "diagnos*"
except the safe "not a diagnosis"), medication directives ("stop/start taking", "increase/decrease
your dose"), causal overclaims ("this caused", "guaranteed", "cure"), and uncited "treatment plan"
(allowed only when attributed to a clinician). Detection is high-recall: false positives are
acceptable, false negatives are not. See `tests/test_safety.py` and `tests/test_narrative.py` for
the adversarial suite (every listed unsafe string must fail).

## Crisis handling

`safety/crisis.py` is a faithful port of the app's `crisis-detection.ts` (tiers + 988/741741/911
resources). It scans **input** text and errs toward showing help. It is NOT a clinical screen.
Resource numbers are US; do not localize without verified per-country numbers.

## Privacy

Raw journal text is never persisted to ML tables or logs — only aggregated, privacy-safe features
leave the boundary. The Supabase service-role key is read from the environment only. Synthetic data
is clearly marked (`persona` column) and never mixed with real data.

## Operating modes

- `rules_only` — default; the ported rule engine only. Used pre-launch and as fallback.
- `ml_assistive` — ML scores are used but heavily abstaining + gated (current target once a model
  is trained and saved).
- `ml_validated` — only after evaluation thresholds (calibration + recall on safety classes) pass.

If model quality is weak for a given outcome (see MODEL_CARD: next-day migraine ranking), ship
rules-only for that outcome and document the limitation.
