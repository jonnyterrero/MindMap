# MindMap ML layer

Safety-first, calibrated, **abstaining** mental-health pattern/risk engine for the MindMap PWA.

This package is intentionally **decoupled** from the Next.js app: no imports either
direction. The two meet only at Supabase. Python trains and runs **daily batch scoring
offline**, writes prediction rows, and the app simply **reads** them — mirroring MindMap's
existing "predictions persisted, app reads" pattern.

## Non-negotiable safety contract

- **Abstention first.** Insufficient/uncertain data → `abstain` ("not enough data yet"),
  never a fabricated number or recommendation.
- **Calibrated uncertainty.** Every prediction ships a confidence/uncertainty value; the eval
  harness measures *calibration* (reliability/ECE), not just accuracy.
- **No diagnosis, no clinical claims, no treatment/medication advice.** Suggestions are gentle,
  optional, non-clinical — never directives.
- **Evidence-grounded only.** A recommendation may surface only if it maps to a curated evidence
  prior or a retrieved research passage.
- **Output gate.** A final guard checks every user-facing string for banned diagnostic phrasing
  and required uncertainty framing; it can downgrade or suppress output.
- Every persisted prediction carries a `model_version`.

## Layout

```
mindmap_ml/
  schema.py        typed dataclasses mirroring Supabase tables (+ range validation)
  config.py        thresholds, paths, model-version constants
  synthetic/       schema-faithful generator + personas with KNOWN ground-truth effects
  features/        declarative feature spec (single source of truth) + pure transforms
  eval/            baseline (faithful port of the TS engines), metrics, harness, reports
  safety/          abstention contract + output-validation gate
  models/          uniform predict(features) -> Prediction interface
  insights/        lagged correlations + multivariate drivers
  evidence/        corpus ingest -> curated priors + local RAG index + retrieve
  narrative/       model + evidence -> guarded, cited text (claude-opus-4-8)
  serving/         batch scoring -> Supabase upsert (all DB I/O isolated here)
tests/             pytest golden/fixture tests per module
```

Models are a uniform interface: `predict(features) -> Prediction{risk, uncertainty,
contributing_factors, abstained, model_version}`. The eval harness consumes any such callable.

## Setup

Requires [uv](https://docs.astral.sh/uv/). No system Python needed — uv manages it.

```bash
cd ml
uv sync            # creates .venv, installs Python 3.11+ and deps
cp .env.example .env
```

## Commands

```bash
uv run pytest                       # tests + golden fixtures
uv run ruff check .                 # lint
uv run mypy                         # type check

uv run python -m mindmap_ml.synthetic.generate              # (P1) sample synthetic data -> data/
uv run python -m mindmap_ml.eval.run                        # (P1) baseline calibration+abstention report
uv run python -m mindmap_ml.eval.compare                    # (P3) ML vs baseline, leave-user-out
uv run python -m mindmap_ml.serving.score_batch --synthetic --dry-run   # (P5) batch scoring (no DB)
```

Optional extras (lazy-imported; not needed for tests):

```bash
uv pip install '.[serving]'    # supabase client for live batch writes
uv pip install '.[narrative]'  # anthropic client for LLM narratives
```

Docs: [SAFETY_POLICY.md](SAFETY_POLICY.md) · [MODEL_CARD.md](MODEL_CARD.md) · [VALIDATION_PLAN.md](VALIDATION_PLAN.md)

## Synthetic → real swap

Pre-launch, everything validates against **synthetic** data with known ground-truth effects.
Switching to real data is a one-file change: `synthetic/generator` → `serving/supabase_io` read.

## Status

Pre-launch, single primary user, validated on synthetic data first. Default operating mode is
**rules_only** until models pass calibration + abstention acceptance in the harness.
