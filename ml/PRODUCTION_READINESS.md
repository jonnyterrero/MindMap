# MindMap ML — Production-Readiness Checklist

Scope: **wellness now, clinical/acquirable later.** Readiness here is mostly
**discipline + provenance**, not model performance — because at current data
volumes the honest product is descriptive (Tier-0), not predictive.

## Operating mode
- [x] Default **`rules_only` / descriptive** (no learned model served).
- [x] Hard promotion gate to `ml_assistive`: a learned model ships only if it beats
      the Tier-0 baseline on **calibration AND ranking** via leave-user-out + walk-forward
      (`eval/walk_forward`, `eval/compare`), thresholds pre-registered in VALIDATION_PLAN.

## Every user-facing output
- [x] Abstention-first ("not enough data yet"); data-sufficiency countdown from the
      power analysis (`synthetic/power`, ~30 logged days before correlations surface).
- [x] "Not a diagnosis" + uncertainty framing; patterns, never causes.
- [x] Evidence-cited recommendations only (`evidence/` priors + retrieval); no uncited advice.
- [x] Output gate on every string (`safety/gate`); banned diagnostic/medical/causal phrasing blocked.
- [x] Crisis routing: PHQ-9 item-9 flag + keyword detection (`safety/crisis`) → resources, bypassing normal output.

## Data & provenance
- [x] Data dictionary + label definitions + lineage (DATA_DICTIONARY, LABELS_AND_INSTRUMENTS).
- [x] Continuous-calendar missingness; label-quality flags.
- [x] Provenance on persisted rows (`model_version`, `source`, `entry_date`) — migration 019.
- [ ] Data-retention + de-identification policy (write before any acquisition conversation).

## Privacy & safety
- [x] Raw journal text never used as an ML input or persisted to ML tables/logs.
- [x] Service-role key from env only; never logged; optional deps lazy-imported.
- [ ] Independent review of the banned-phrase gate + crisis routing before clinical use.

## Reproducibility
- [x] Pinned deps (`uv.lock`), seeded synthetic, deterministic tests.
- [x] Pre-registered acceptance thresholds (VALIDATION_PLAN); model card kept current.

## Explicitly NOT ready / do not claim
- Predictive ML performance, clinical validity, or population generalization.
- Anything trained on synthetic data.

## To activate the learned layer later (when data accrues)
~30–50 users × ≥6–8 weeks (≥~50–100 positive events/outcome pooled) → train + save a
model artifact → walk-forward/leave-user-out must beat Tier-0 → wire a daily cron →
surface enriched rows in the app. Until then: ship Tier-0, harvest clean data.
