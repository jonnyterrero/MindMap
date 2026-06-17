# Verified Mindmap Generation — 3-Stage Pipeline Design

Turn free-form user text (journal / brainstorm / notes) into a **verified concept
graph**. The generator proposes; an **external, multi-component verifier disposes**.
Only verified claims are returned; everything else is downgraded, abstained, or
suppressed (**fail-closed**). Built for **data scarcity**: off-the-shelf foundation
models + small trainable components + weak supervision + synthetic + human gold.

Non-negotiables: validation is external to generation · **agreement ≠ validation** ·
plausibility ≠ evidence · only verified info returns · every element carries
provenance (output → spans → step → model/version/rule) · uncertainty is explicit.

## Pipeline

```
RAW TEXT
  └─▶ STAGE 1  Import & Digest        (deterministic)   [IMPLEMENTED: graph/ingest.py]
        normalize → segment → CHAR-OFFSET spans (reversible to raw)
  └─▶ STAGE 2  Structured Generation  (probabilistic)   [IMPLEMENTED: graph/generate.py]
        foundation LLM (claude-opus-4-8, strict JSON) → candidate concepts/nodes/edges,
        each citing span_ids + self-confidence  (CANDIDATES ONLY)
  └─▶ STAGE 3  External Verification  (hybrid)          [IMPLEMENTED: graph/verify.py]
        (A) schema validator [det, fail-closed] (B) provenance checker [det]
        (C) NLI/entailment grounder [SEPARATE model] (D) graph consistency [rules+emb]
        (E) retrieval-evidence scorer [reuse evidence/] (F) calibrator → claim_class
        (G) human escalation
        accept | downgrade | abstain | reject
  └─▶ VERIFIED MINDMAP ARTIFACT (+ suppressed[] + lineage) → Supabase → app reads
```
Generator cannot self-approve. Verifier is a different model/objective. Claims are
classed `directly_supported | weakly_inferred | unverifiable`.

## Stage contracts

| Stage | Purpose | In | Out | Type | Versioned | Measured |
|---|---|---|---|---|---|---|
| 1 | digest → addressable spans | raw text | RawDocument + TextSpan[] | deterministic | normalizer, segmenter | offset round-trip exactness (100%), seg F1, coverage |
| 2 | propose candidate graph | spans | candidate Concept/Node/Edge + provenance | probabilistic/hybrid | prompt hash, model id, taxonomy, dedup thr | candidate node/edge P/R, hallucination rate, span-citation validity |
| 3 | verify & decide | candidate graph + spans | VerifierDecision[] + verified artifact | hybrid | NLI model, thresholds, calibrator, rules, priors hash | true/false-accept, false-reject, graph validity, provenance completeness, ECE/Brier, contradiction P/R |

### Stage 1 offset invariant (load-bearing)
`raw_text` is **NFC-canonical**; `raw_start/raw_end` index into it. `norm_text` is the
normalized view; `norm_text[norm_start:norm_end] == span.text` exactly. Reversibility:
re-normalizing `raw_text[raw_start:raw_end]` reproduces `span.text`. The norm→raw map is
strictly increasing. If offsets drift, all downstream provenance is void — hence tested
aggressively in `tests/test_graph_ingest.py`.

## Data schema (Stage 1 concrete; later stages sketched)
See `graph/schema.py` for the implemented `RawDocument` + `TextSpan`. Stage-2/3 objects
(`Concept`, `Node`, `Edge`, `EvidenceLink`, `VerifierDecision`, `Confidence`,
`MindmapArtifact`, `AuditLog`) land with their stages — full field list in the design
delivered with this pipeline. Every claim carries `evidence: span_id[]` (or an explicit
`inferred` marker + type + confidence), `claim_class`, `provenance`, and a verifier decision.

## Validation rules (required now → enforced in Stage 3)
1. Schema invalid → reject (fail-closed). 2. Every node/edge maps to ≥1 span **or** is
marked inferred (type+confidence). 3. Cited spans must exist + round-trip. 4. `directly_supported`
needs NLI=entail ≥ τ_high. 5. Inferred edges carry inference_type + calibrated confidence
(causal confidence capped). 6. Contradictions flagged, never silently merged. 7. Unverifiable
psychological/causal claims with no span + no prior → reject/down-rank. 8. Hallucinated
entities/themes/links → suppress. 9. Near-duplicate nodes merged (embedding). 10. Coverage
sanity for under/over-splitting.

## Model roles
Stage 1: `spaCy`/regex + heuristic lang-detect (rules). Stage 2: `claude-opus-4-8` extractor
(few-shot, strict JSON) + embedding dedup (sentence-transformers or reuse `evidence/index.py`)
+ relation typer. Stage 3: **separate** NLI cross-encoder (DeBERTa-MNLI class) or adversarial
LLM verifier + deterministic graph/schema checks + retrieval (reuse `evidence/`) + isotonic/Platt
calibrator + `safety/gate.py`/`safety/contract.py` + human queue.

## Training & evaluation (data-limited)
Off-the-shelf for extraction/NLI/embeddings. Train only small components: calibrator, dedup
threshold, relation classifier (100–300 gold). Weak supervision (connectives) = silver, never
test. Synthetic (reuse `synthetic/`) for pipeline tests + threshold setting only — **not** the
gold accept set (circularity). **Human gold (~200 entries) is mandatory.** Splits: by-user
(leave-user-out) + temporal + a **challenge set** (sarcasm, metaphor, contradiction, fragments,
emotional, low-context). Reuse `eval/metrics.py` (ECE/Brier/reliability) + `eval/harness`.

## Scoring metrics
Span boundary P/R/F1; node/edge P/R/F1 (causal edges separately); **verifier true-accept /
false-accept / false-reject** (false-accept is the primary safety metric); graph validity rate;
provenance completeness; contradiction P/R; **ECE + Brier**; abstention coverage; per-challenge
rare-case slices; error taxonomy counts. Release gates: false-accept ≤ target, ECE ≤ target,
graph validity = 100%, provenance completeness = 100% for `directly_supported`.

## Failure modes (abridged — see full design)
ambiguous/fragmented/sarcasm/contradiction/emotional-over-interpretation/paraphrase-dup/
weak-causal/category-collapse/over-undersplit/spurious-edges/calibration-drift/verifier-over-
&under-rejection/distribution-shift — each with cause, detection signal, mitigation, and the
metric that reveals it. Common thread: **fail closed and abstain** rather than emit unsupported.

## Final output contract
App receives a `MindmapArtifact` with only verified/downgraded nodes+edges, each with
calibrated confidence + claim_class + provenance (clickable to source spans); `suppressed[]`
lists rejected items with reason codes (transparency); `abstained=true` when evidence is too
weak; coverage + calibration metadata + lineage. On error: return the partial verified subset
or abstain — never raw generator output. Persisted to Supabase (new tables, RLS user + provider
read), app reads (same pattern as `mindmap_predictions`/`mindmap_ml_summaries`).

## Implementation notes
Thresholds set on validation, versioned, dashboarded (default conservative: favor false-reject).
Fallbacks: LLM fail → rule skeleton; NLI down → retrieval+rules with capped confidence; any
verifier component error → fail-closed. HITL queue on `[τ_low, τ_high)` / contradiction / safety,
feeding the gold set. Batch eval via harness; online shadow + sampled audits + drift monitors.
Versioning: content-hash prompts, semver models/schema/rules/calibrators; every artifact stores
`pipeline_version` + per-component `model_version`.

## Status
- **Stage 1 — IMPLEMENTED**: `graph/ingest.py` + `graph/schema.py`, offset-integrity tested
  (`tests/test_graph_ingest.py`). Deferred: markdown stripping, clause units, ML/`spaCy`
  segmentation, real language detection.
- **Stage 2 — IMPLEMENTED**: `graph/generate.py` — injectable LLM extractor (claude-opus-4-8,
  strict JSON, span-cited candidates) + deterministic rule-skeleton fallback + TF-IDF dedup.
  Deferred: trained relation classifier, sentence-transformer dedup, self-consistency sampling.
- **Stage 3 — IMPLEMENTED**: `graph/verify.py` — schema + provenance + entailment (injectable;
  default = conservative lexical placeholder) + graph consistency (dangling/contradiction) +
  rule calibrator; fail-closed. `graph/pipeline.py` orchestrates Stage 1→2→3. Tested:
  `tests/test_graph_{generate,verify,pipeline}.py` incl. agreement≠validation + fail-closed.
  **Deferred (required before trusting outputs): a real NLI cross-encoder / adversarial-LLM
  verifier, a trained calibrator, and the ~200-entry human gold + challenge set + eval harness.**
- **Persistence — planned**: `serving/` writer + Supabase migration; app read-only mindmap view.
