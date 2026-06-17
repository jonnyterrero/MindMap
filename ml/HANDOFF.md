# MindMap+ ML — Session Handoff

_Last updated: 2026-06-17 · `main` @ `ace7059` · working tree clean · 173 `ml/` tests passing, ruff + mypy clean._

This doc orients a fresh session. There are **two independent ML workstreams** in `ml/`.
Read this, then the canonical specs it points to.

---

## 0. Orientation & gotchas (read first)

- **Git root** is `C:\Users\JTerr\OneDrive\Programming Projects\Mindmap+\MindMap` (has `frontend/`, `supabase/`, `ml/`). Sessions often open in the empty scratch dir `MindMap-2` — **don't build there.** Use absolute paths or `git -C <root>`.
- **All ML code lives in `ml/`** — a self-contained Python package, **decoupled** from the Next app (they meet only at Supabase). Run commands from `ml/`.
- **Tooling = `uv`** (no system Python on PATH; uv manages 3.11+). Standard gate:
  ```bash
  cd ml
  uv run pytest          # 173 tests (~50s; the forecast-comparison test is the slow one)
  uv run ruff check .
  uv run mypy
  ```
- **Cadence used throughout:** branch off `main` → implement test-first → full gate green → `--no-ff` merge to `main` → push → delete branch. Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Windows notes:** CRLF warnings on `git add` are harmless. The auto-mode classifier **blocks secrets on the command line** and `git branch -D` (use `git fetch --prune` then `-d`). Don't read `.env`/credential files.
- **Memory files** (auto-loaded next session) live in the Claude memory dir: `ml-layer-build.md`, `mindmap-graph-pipeline.md`, `mindmap-stack-and-connections.md`. They mirror this doc.

---

## 1. Workstream A — Health pattern/risk ML layer (DONE / LIVE)

Predicts next-day risk + descriptive insights from numeric daily check-ins. **Complete, merged, deployed; running in `rules_only`** because prod has too little data to train (~10 entries / 2 users).

- Phases 0–6 (schema, synthetic personas, features, ported rule baseline, calibrated forecaster, evidence RAG, batch serving, guarded LLM narrative) + a **Tier-0 descriptive layer** (calendar/adherence, conditional base rates, naive forecasts, walk-forward eval, power analysis, PHQ-9/GAD-7, **clinician summary**).
- **Surfaced in the app**: `mindmap_ml_summaries` (migration 020, applied to prod) ← Python batch writes ← app reads via `frontend/app/(app)/insights/`. Migration 019 (extends `mindmap_predictions`) also applied.
- **Daily cron**: `.github/workflows/ml-summary-cron.yml` (GitHub Action; **inert until repo secrets `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` are set** — note: GitHub Actions secrets, NOT Vercel).
- Docs: `MODEL_CARD.md`, `SAFETY_POLICY.md`, `VALIDATION_PLAN.md`, `DATA_DICTIONARY.md`, `LABELS_AND_INSTRUMENTS.md`, `PRODUCTION_READINESS.md`.
- **Remaining (future, data-gated):** train the learned forecaster once ~30–50 users × 6–8 wks accrue (it's parked in `models/forecast.py`, must beat Tier-0 on walk-forward calibration+ranking before flipping to `ml_assistive`); replace seed evidence corpus with real citations.

⚠️ **Action item:** the user pasted prod Supabase/Anthropic keys in chat earlier — **rotate them**, then use the rotated service-role key as the GitHub Actions secret.

---

## 2. Workstream B — Verified mindmap pipeline (ACTIVE — pick up here)

Turns free-form user text (journal/brainstorm/notes) into a **verified concept graph**. Generation is separated from verification; only verified claims return (**fail-closed**). Canonical spec: **`ml/MINDMAP_PIPELINE_DESIGN.md`**. All code in `ml/mindmap_ml/graph/`.

### What's built (all merged, tested)
| Stage | File | What it does |
|---|---|---|
| 1 Import & digest | `graph/ingest.py` | NFC canonicalize → normalize w/ strictly-increasing norm→raw offset map → sentence/fragment segmentation → `TextSpan[]`. **Offset integrity is the load-bearing invariant** (re-normalizing `raw_text[raw_start:raw_end]` reproduces `span.text`). |
| 2 Generation | `graph/generate.py` | Injectable LLM extractor (`claude-opus-4-8`, strict JSON, candidates cite span_ids) + deterministic **rule-skeleton fallback** + TF-IDF dedup. Candidates only — can't self-approve. |
| 3 Verification | `graph/verify.py` | **External** verifier: schema + provenance + entailment grounder + graph-consistency (dangling/contradiction) + rule calibrator. Decisions: accept / downgrade / abstain / reject. Two grounders behind the `Entailment` protocol: `LexicalEntailment` (conservative default, no deps) and `LLMEntailment` (adversarial, injectable, fail-closed). |
| orchestrator | `graph/pipeline.py` | `run_pipeline(text, user_id=...) -> MindmapArtifact` (Stage 1→2→3). Everything injectable → runs offline/deterministic. |
| schema | `graph/schema.py` | RawDocument, TextSpan, Node, Edge, CandidateGraph, Confidence, VerifierDecision, MindmapArtifact. |
| eval | `graph/gold.py` + `graph/evaluate.py` | Hand-authored 16-claim gold/challenge set + harness measuring TA/**false-accept**/FR/TR, P/R/F1, per-category, Brier/ECE. CLI: `uv run python -m mindmap_ml.graph.evaluate`. |

Tests: `tests/test_graph_{ingest,generate,verify,pipeline,evaluate}.py`. They prove **agreement ≠ validation** (a 0.99-confidence hallucination is suppressed), fail-closed causal/contradiction/provenance handling, and end-to-end provenance traceability.

### Measured baseline (lexical grounder, 16-claim gold)
```
recall 1.0 · false-accept rate 0.25
0 false-accepts on: hallucination, contradiction ×2, emotional over-interpretation,
                    low-context over-claiming, psychological interpretation
false-accepts:      metaphor + sarcasm   ← shallow grounding's limit (the target to fix)
```

### Invariants to respect (do not regress)
1. **Offset integrity** (Stage 1) — if span offsets drift, all provenance is void. Heavily tested.
2. **Agreement ≠ validation** — the verifier must stay a *separate objective* from the generator; never "approve because the generator was confident."
3. **Fail-closed** — unsupported/contradictory/anonymous/invalid claims are downgraded or suppressed, never emitted.
4. Everything **injectable** (LLM client, entailment grounder) so tests run with no key/network.

### Remaining work (prioritized) — this is where to continue
1. **Wire a real grounder in prod.** `LLMEntailment` exists but needs a real client (Anthropic, lazy — mirror `narrative/compose.py`'s `AnthropicClient`); or a cross-encoder NLI (DeBERTa-MNLI — heavier, needs torch). Acceptance: figurative false-accepts (metaphor + sarcasm) drop in `graph/evaluate` while recall stays high.
2. **Expand the gold set** to ~200 dual-annotated real entries (the mandatory human spend). Release gate = **false-accept ≤ target**, graph-validity = 100%, provenance-completeness = 100% for `directly_supported`, ECE ≤ target. Add edge-level gold (current gold is node-level).
3. **Retrieval-evidence scorer** for psychological/causal claims — reuse `evidence/` (index + curated priors) so such claims must map to a prior or be down-ranked.
4. **Trained calibrator** (isotonic/Platt on gold; reuse `eval/metrics.py` ECE/Brier) to replace the rule_v0 calibrator.
5. **Persistence + app surface** (the "make it reach users" step, deferred as the larger/riskier option): new Supabase migration (`mindmap_graphs` table, RLS user-own + provider read, mirroring migration 020) + a `serving/` writer + a read-only "Mindmap" view in `frontend/app/(app)/`. This involves a prod migration + TS + deploy.

---

## 3. Reuse map (don't rebuild these)
`evidence/*` (ingest/index/retrieve + curated priors) → Stage-3 retrieval + provenance corpus · `safety/gate.py`+`contract.py` → rules + abstention · `eval/metrics.py` (ECE/Brier/AUROC/reliability) + harness → calibration/scoring · `serving/supabase_io.py` + a new migration → persistence · `synthetic/` → stress data with known ground truth · `narrative/compose.py` → the pattern for a lazy Anthropic client.

## 4. Suggested first action next session
`cd ml && uv run pytest` to confirm green, then `uv run python -m mindmap_ml.graph.evaluate` to see the verifier baseline. Then pick remaining-work item #1 (real grounder) or #5 (persistence + app surface) depending on whether you want correctness or reach next.
