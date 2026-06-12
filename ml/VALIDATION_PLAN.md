# MindMap ML — Validation Plan

## Principle

Pre-launch there is no real data, so validation runs against **synthetic personas with known
ground-truth effects** (`synthetic/scenarios.py`). A model is only promoted if it beats the rule
baseline on **error AND calibration** without worse abstention.

## Splits

- **Leave-user-out** (implemented, `eval/compare.py`): ML trains on one set of users and is judged
  on users it never saw.
- **Temporal** (per-user as-of evaluation): the harness only ever feeds a model past+current entries
  to predict a future label — no leakage.
- **Cold-start**: thin-history points are included; the model must abstain rather than guess.

## Metrics (`eval/metrics.py`)

Discrimination: AUROC, AUPRC, precision/recall/F1 at threshold. **Calibration: ECE + reliability
curve, Brier.** **Abstention: coverage** (and metrics computed on the covered subset only). Calibration
and abstention are first-class — a confident-but-wrong model fails.

## Acceptance gates

- No model is "ml_validated" without calibration (low ECE) on its outcome.
- A model only replaces the baseline for an outcome if AUROC ≥ baseline **and** ECE ≤ baseline.
- Safety/crisis classifiers must have high recall before use (false negatives are the dangerous error).
- If a model is weak for an outcome, ship `rules_only` for that outcome and document it (see
  MODEL_CARD: next-day migraine).

## Reproduce

```bash
uv run python -m mindmap_ml.eval.run        # baseline calibration + abstention report
uv run python -m mindmap_ml.eval.compare    # ML vs baseline (leave-user-out)
uv run pytest                               # all golden/acceptance tests
```

## Before clinical / public release

Replace synthetic validation with real, consented data; re-run all splits; replace the seed evidence
corpus with peer-reviewed sources; add subgroup/fairness reporting; independent safety review of the
banned-phrase gate and crisis routing.
