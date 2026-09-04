# MindMap Real ML Training + Production Hosting Handoff

**Project:** MindMap  
**Repository:** `jonnyterrero/MindMap`  
**Purpose:** Engineering handoff for Cursor / Claude Code  
**Status:** Existing ML scaffold is substantial; next phase is real-data correctness, training infrastructure, validation, and production hosting.

---

# 1. Executive Summary

MindMap already contains a legitimate ML foundation. The next step is **not to invent a new model from scratch**.

The current repository already includes:

- a real scikit-learn forecasting model
- deterministic feature engineering
- leave-user-out evaluation
- calibration metrics
- abstention behavior
- rules-based fallback
- Supabase data ingestion
- batch prediction persistence
- model versioning
- safety gates
- synthetic test data
- production-readiness documentation
- a verified graph pipeline for journal-derived concept mapping

The priority now is to turn this scaffold into a trustworthy real-data ML lifecycle:

```text
real consented MindMap data
        ↓
canonical dataset builder
        ↓
calendar-correct labels
        ↓
train / calibrate / validate / test
        ↓
versioned model artifact
        ↓
shadow inference
        ↓
prospective outcome reconciliation
        ↓
promotion gate
        ↓
production ML-assisted inference
```

Desktop/mobile clients should remain **thin clients**. They should not train models, own model artifacts, or contain privileged infrastructure credentials.

Recommended production architecture:

```text
Next.js / Capacitor clients
        ↓
Supabase Auth + PostgreSQL + RLS
        ↓
AWS ML plane
    ├─ S3 / KMS
    ├─ SageMaker Training
    ├─ SageMaker Model Registry
    ├─ ECR
    ├─ ECS Fargate batch inference
    ├─ EventBridge scheduling
    ├─ Secrets Manager
    └─ CloudWatch
        ↓
mindmap_predictions / mindmap_ml_summaries
        ↓
app reads results
```

The immediate next engineering branch should be:

```text
ml/real-data-correctness
```

Do not start the AWS deployment until the real-data labeling and feature pipeline are correct.

---

# 2. Existing Repo ML Architecture

The active ML package lives under:

```text
ml/
```

Important existing files include:

```text
ml/HANDOFF.md
ml/MODEL_CARD.md
ml/SAFETY_POLICY.md
ml/VALIDATION_PLAN.md
ml/PRODUCTION_READINESS.md
ml/DATA_DICTIONARY.md
ml/LABELS_AND_INSTRUMENTS.md
ml/MINDMAP_PIPELINE_DESIGN.md

ml/mindmap_ml/models/forecast.py
ml/mindmap_ml/features/spec.py
ml/mindmap_ml/features/engineering.py
ml/mindmap_ml/features/calendar.py
ml/mindmap_ml/eval/compare.py
ml/mindmap_ml/serving/score_batch.py
ml/mindmap_ml/serving/supabase_io.py
```

The package uses Python 3.11+, `uv`, NumPy, pandas, scikit-learn, pytest, Ruff, and mypy.

The current learned forecaster is:

```text
imputer
→ standard scaler
→ L2 logistic regression
→ sigmoid calibration
→ lightweight user-specific base-rate adaptation
→ abstention
```

This should remain the first production candidate. Do not replace it with a neural network yet.

---

# 3. Current Prediction Targets

The current system uses operational app-risk labels rather than clinical diagnoses.

Current targets include:

| Prediction Type | Operational Label |
|---|---|
| Migraine | migraine occurs |
| Anxiety | self-reported anxiety >= configured threshold |
| Mood | low mood / elevated depression proxy |
| Pain flare | high migraine intensity proxy |

These labels must remain framed as operational self-tracking targets, not medical diagnoses.

Do not build user-facing predictions such as:

- probability of bipolar disorder
- probability the user "has depression"
- diagnostic ADHD prediction
- medication response recommendation

Safer future targets include:

- next-day migraine event
- next-day high self-reported anxiety
- next-day low-mood event
- next-day low-focus event
- next-day elevated activation trend
- severe migraine event

---

# 4. Critical Real-Data Correctness Issues to Fix First

## 4.1 "Next-day" currently behaves like "next logged entry"

The existing forward-label logic shifts across logged rows.

Conceptually:

```text
log n
→ log n+1
```

That is not always:

```text
calendar day t
→ calendar day t+1
```

Example:

```text
Monday    logged
Tuesday   missing
Wednesday missing
Thursday  logged
```

If Monday's prediction target is defined using the next logged entry, Thursday can incorrectly become the "next-day" target.

This must be corrected before training a production candidate.

### Required fix

Build labels on a continuous calendar:

```text
calendar day t
→ calendar day t+1
```

Rules:

- if the next calendar day's outcome is observed, use it
- if the next calendar day was not logged, mark the target as unknown / NaN
- never treat a missing next-day log as a negative event
- never silently skip missing calendar days during label creation

Use the existing continuous-calendar utilities in:

```text
ml/mindmap_ml/features/calendar.py
```

as the starting point.

---

# 5. Calibration Split Must Become Longitudinally Safe

For real health-related longitudinal data, calibration should not randomly mix nearby rows from the same user.

Use explicit partitions:

```text
TRAIN
    ↓
fit base model

CALIBRATION
    ↓
fit probability calibration

VALIDATION
    ↓
select thresholds / acceptance criteria

LOCKED TEST
    ↓
final untouched performance estimate
```

Required split constraints:

- split by user where possible
- preserve time ordering
- no future information in feature computation
- no row-level random split across the same user's timeline
- no calibration leakage
- no validation/test reuse during model tuning

Recommended evaluation modes:

```text
leave-user-out
temporal holdout
walk-forward validation
cold-start evaluation
prospective shadow evaluation
```

---

# 6. Production Data Reader Must Use an Explicit Allowlist

The ML data reader should not use `select("*")` for production training or inference.

Build an explicit ML-safe field allowlist.

Conceptual allowlist:

```text
user_id
entry_date
sleep_minutes
sleep_quality
hrv
mood_valence
anxiety
depression
mania
focus
productivity
therapy_minutes
outside_minutes
migraine
migraine_intensity
migraine_aura
```

Related-table features should be aggregated separately:

```text
body_pain
med_adherence_rate
routine_completion_rate
pressure
pressure_change
humidity
temp_max
```

Explicitly exclude:

```text
notes
raw journal text
provider notes
names
emails
therapy notes
free-form PHI not required by the model
unnecessary account/profile metadata
```

Also filter:

```text
deleted_at IS NULL
```

and require the correct training-consent state.

---

# 7. Build One Canonical Training Dataset Function

Create one function or module that becomes the single source of truth:

```python
build_training_dataset(...)
```

Its output must be one row per user per actual calendar day.

The dataset builder should:

1. read only explicitly approved fields
2. exclude soft-deleted records
3. filter by training consent
4. normalize dates/timezones
5. create continuous daily calendars
6. join related tables by user/date
7. aggregate body sensations
8. aggregate medication adherence
9. aggregate routine completion
10. join weather/environment signals
11. preserve missingness
12. compute operational labels
13. record label availability / quality
14. output a deterministic schema
15. produce a dataset manifest / hash

Suggested output columns:

```text
user_id
entry_date

sleep_minutes
sleep_quality
hrv

mood_valence
anxiety
depression
mania
focus
productivity

migraine
migraine_intensity
migraine_aura

therapy_minutes
outside_minutes

body_pain
med_adherence_rate
routine_completion_rate

pressure
pressure_change
humidity
temp_max

logged
label_migraine
label_anxiety
label_low_mood
label_low_focus
label_severe_migraine

label_quality_*
```

---

# 8. Keep the First Real Model Simple

## Model 1 — canonical learned baseline

Use the existing approach:

```text
median imputation
→ standardization
→ regularized logistic regression
→ calibration
```

Advantages:

- explainable
- relatively data-efficient
- inexpensive
- easy to debug
- easy to version
- easy to calibrate
- cheap to run in batch
- appropriate for structured tabular data
- suitable for CPU-only hosting

## Model 2 — challenger

Once the entire pipeline works, introduce a tree-based challenger such as:

```text
XGBoost / LightGBM / HistGradientBoosting
```

Do not automatically promote it.

Promotion should depend on:

```text
calibration
+ discrimination
+ false positive behavior
+ false negative behavior
+ abstention behavior
+ stability
+ interpretability
```

---

# 9. Feature Count Must Be Controlled

The current feature space is already large relative to the likely size of the first real dataset.

Create a smaller `Core Feature Spec v1`.

Recommended initial signals:

```text
sleep_minutes
sleep_quality
anxiety
mood_valence
focus
migraine
migraine_intensity
hrv
body_pain
med_adherence_rate
routine_completion_rate
```

Recommended temporal transforms:

```text
lag 1
lag 2
rolling mean 3
rolling mean 7
rolling std 7
delta 1
missingness flag
```

Only expand once data volume supports it.

---

# 10. Training Consent Must Be Separate From AI Feature Consent

Do not treat:

```text
AI insights enabled
```

as equivalent to:

```text
permission to use data for population-level model training
```

Create explicit concepts such as:

```text
allow_ai_insights
allow_personalized_ml
allow_deidentified_model_training
```

Store:

```text
consent type
consent version
timestamp
purpose
revocation timestamp
```

Training-data extraction must enforce consent.

---

# 11. Real Model Training Lifecycle

```text
CONSENTED REAL DATA
        │
        ▼
Canonical dataset snapshot
        │
        ├── dataset version
        ├── schema version
        ├── feature spec version
        └── hash / lineage
        │
        ▼
User/time-aware split
        │
        ├── train
        ├── calibration
        ├── validation
        └── locked test
        │
        ▼
Candidate model
        │
        ▼
Evaluation
        │
        ├── AUROC
        ├── AUPRC
        ├── Brier
        ├── ECE
        ├── sensitivity
        ├── specificity
        ├── false positive rate
        ├── false negative rate
        └── abstention coverage
        │
        ▼
Promotion gate
        │
    ┌───┴────┐
    │        │
   fail     pass
    │        │
 discard   registry
             │
             ▼
        shadow mode
             │
             ▼
 prospective outcomes
             │
             ▼
        final promotion
```

Promotion should occur **per outcome**, not globally.

---

# 12. Shadow Mode Is Mandatory

The first real model should not immediately replace the rule engine.

Run:

```text
rules prediction → user-facing
ML prediction    → internal shadow row
```

Later, once the target day is available:

```text
prediction
+
actual future self-report
→ prospective evaluation record
```

Measure:

- AUROC
- AUPRC
- Brier score
- ECE
- sensitivity
- specificity
- false positive rate
- false negative rate
- abstention coverage
- per-user performance
- calibration by cohort
- drift over time

The learned model should only become user-facing after acceptable prospective performance.

---

# 13. Automatic Outcome Reconciliation

User feedback like `accurate` / `inaccurate` is useful but should not be the primary ground truth.

Example:

```text
September 4:
model predicts P(high anxiety tomorrow) = 0.71

September 5:
user logs anxiety = 8

automatic outcome:
label_high_anxiety = 1
```

If September 5 is not logged:

```text
outcome = unknown
```

Never:

```text
outcome = false
```

Build:

```python
reconcile_prediction_outcomes(...)
```

that:

1. finds matured prediction windows
2. looks for observed future data
3. computes operational outcome labels
4. marks missing outcomes as unknown
5. stores evaluation-ready prediction/outcome pairs
6. never modifies the original prediction value

---

# 14. Minimum Data Gate

Do not train or promote a model just because some rows exist.

Use the current repo's rough engineering gate as a minimum:

```text
~30–50 users
×
6–8 weeks of logging
```

and enough positive events per target.

The limiting quantity is positive outcome events, not total rows.

Track:

```text
total users
users with >= 7 days
users with >= 30 days
users with >= 42 days
total logged days
migraine events
high anxiety events
low mood events
low focus events
severe migraine events
```

These are engineering readiness thresholds, not clinical validation.

---

# 15. Infrastructure Decision

## Recommended provider

Use AWS for the ML plane while keeping the application itself on the current Next.js + Supabase architecture for now.

Do not migrate the entire application solely because ML compute needs a home.

| Component | Platform |
|---|---|
| Web/mobile UI | Next.js + Capacitor |
| Authentication | Supabase |
| User data source of truth | Supabase PostgreSQL |
| Row-level access control | Supabase RLS |
| Training snapshots | AWS S3 |
| Encryption keys | AWS KMS |
| Training compute | SageMaker Training |
| Model artifacts | S3 |
| Model registry | SageMaker Model Registry |
| Container registry | ECR |
| Batch inference | ECS Fargate |
| Batch scheduling | EventBridge |
| Secrets | Secrets Manager |
| Monitoring | CloudWatch |
| Prediction persistence | Supabase |
| Model execution on device | Never initially |

---

# 16. Why Batch Inference First

MindMap's core models predict daily behavioral/symptom trends. There is no reason to keep a model server alive 24/7 initially.

Use:

```text
EventBridge schedule
        ↓
ECS Fargate task
        ↓
load current promoted model
        ↓
pull ML-safe structured user data
        ↓
construct feature rows
        ↓
score eligible users
        ↓
apply abstention + safety
        ↓
write prediction rows to Supabase
        ↓
task exits
```

This maps well to the existing:

```text
ml/mindmap_ml/serving/score_batch.py
```

architecture.

---

# 17. Future Near-Real-Time Inference

Later:

```text
check-in saved
        ↓
Next.js server action / internal backend
        ↓
authenticated server-to-server request
        ↓
prediction service
        ↓
prediction persisted
        ↓
UI revalidates
```

Possible implementation:

```text
API Gateway
→ Lambda
→ small sklearn artifact
```

or later:

```text
SageMaker real-time endpoint
```

Do not call prediction infrastructure directly from the browser/mobile client.

---

# 18. Never Put Privileged ML Infrastructure on the Client

Desktop/mobile/PWA clients should never receive:

```text
AWS credentials
Supabase service-role key
model training credentials
cross-user datasets
model registry credentials
full training snapshots
privileged prediction endpoints
```

The client should only:

```text
authenticate
submit the user's own data
read allowed predictions
display safe results
```

---

# 19. Training Artifact Contract

Every trained candidate should produce:

```text
model artifact
metrics report
dataset manifest
feature manifest
training config
model card update
```

Suggested structure:

```text
artifacts/
  forecast/
    v3.0.0/
      model.joblib
      metrics.json
      dataset_manifest.json
      feature_manifest.json
      training_config.json
      model_card.json
```

`dataset_manifest.json` should include:

```text
dataset version
snapshot timestamp
row count
user count
date range
event counts per target
consent policy version
schema version
dataset hash
```

`feature_manifest.json` should include:

```text
feature names
feature ordering
feature spec version
required raw signals
missingness behavior
normalization behavior
```

---

# 20. Model Registry

Registry states:

```text
candidate
shadow
approved
production
retired
rejected
```

Each model version should track:

```text
model version
training dataset version
feature spec version
code commit
metrics
promotion decision
promotion timestamp
rollback artifact
```

---

# 21. Safety Architecture Must Stay in Front of User Outputs

Preserve:

```text
model output
    ↓
abstention
    ↓
confidence / uncertainty
    ↓
safety gate
    ↓
evidence / contributing factors
    ↓
safe user-facing explanation
```

Do not use:

```text
user data
→ LLM
→ medical-risk prediction
```

Use:

```text
structured data
→ validated statistical/ML model
→ numerical result
→ guarded explanation layer
```

The LLM may explain a validated result but must not invent the risk score.

---

# 22. Development Branch Plan

## Branch 1 — `ml/real-data-correctness`

Objectives:

- calendar-day labels
- missing-day handling
- explicit ML field allowlist
- soft-delete filtering
- training-consent filtering
- canonical daily dataset assembler
- data quality report
- tests

Acceptance criteria:

```text
next-day means actual calendar t+1
missing future log != negative target
notes never enter ML dataset
deleted rows never enter ML dataset
non-consented users never enter population training dataset
feature builder deterministic
tests green
```

## Branch 2 — `ml/real-training-pipeline`

Objectives:

- explicit train/calibration/validation/test split
- user-aware + time-aware splitting
- training CLI
- model artifact save/load
- feature manifest
- dataset manifest
- metrics report
- candidate comparison
- model promotion decision report

Acceptance criteria:

```text
no user/time leakage
deterministic seeded training
reproducible artifact
metrics produced per outcome
artifact load test passes
baseline comparison generated
no automatic production promotion
```

## Branch 3 — `ml/shadow-evaluation`

Objectives:

- shadow prediction rows
- prediction window fields
- automatic outcome reconciliation
- prospective metrics
- per-outcome promotion state

Acceptance criteria:

```text
user does not see shadow predictions
outcomes attach only after prediction time
missing outcomes remain unknown
prospective metrics can be computed
historical prediction values remain immutable
```

## Branch 4 — `infra/aws-ml-plane`

Objectives:

- S3 model/data buckets
- KMS
- ECR
- training container
- SageMaker training job
- model registry
- Fargate scoring task
- EventBridge schedule
- Secrets Manager
- CloudWatch
- IAM least privilege
- CI/CD integration

Acceptance criteria:

```text
training job can run without local machine
model artifact lands in controlled storage
Fargate can load promoted artifact
Fargate can read only required Supabase data
predictions write back successfully
no secrets committed
monitoring/logging enabled
rollback path documented
```

---

# 23. Recommended Current Data-Readiness Query

```sql
WITH user_stats AS (
    SELECT
        user_id,
        COUNT(*) AS logged_days,
        MIN(entry_date) AS first_day,
        MAX(entry_date) AS last_day,
        COUNT(*) FILTER (WHERE migraine = true) AS migraine_days,
        COUNT(*) FILTER (WHERE anxiety >= 7) AS high_anxiety_days,
        COUNT(*) FILTER (
            WHERE mood_valence < 0 OR depression >= 6
        ) AS low_mood_days,
        COUNT(*) FILTER (
            WHERE focus <= 3
        ) AS low_focus_days
    FROM public.mindmap_entries
    WHERE deleted_at IS NULL
    GROUP BY user_id
)
SELECT
    COUNT(*) AS total_users,
    COUNT(*) FILTER (WHERE logged_days >= 7) AS users_7_plus,
    COUNT(*) FILTER (WHERE logged_days >= 30) AS users_30_plus,
    COUNT(*) FILTER (WHERE logged_days >= 42) AS users_42_plus,
    SUM(logged_days) AS total_logged_days,
    SUM(migraine_days) AS migraine_events,
    SUM(high_anxiety_days) AS high_anxiety_events,
    SUM(low_mood_days) AS low_mood_events,
    SUM(low_focus_days) AS low_focus_events
FROM user_stats;
```

Do not expose individual user data in readiness reports.

---

# 24. What Not to Do

Do not:

- train a production model on synthetic data
- treat one user's history as population ground truth
- jump directly to PyTorch
- use random row-level train/test splitting
- treat a missing future log as a negative outcome
- include journal text in the forecasting model
- ship ML predictions before prospective shadow evaluation
- let an LLM generate numerical risk predictions
- put model infrastructure credentials in mobile/web clients
- load arbitrary untrusted pickle/joblib artifacts
- claim clinical validity
- claim diagnostic performance
- claim HIPAA compliance merely because infrastructure services can support HIPAA workloads
- migrate the entire application to AWS before the ML plane requires it

---

# 25. First Model Recommendation

Keep `ForecastModel` as the first real production candidate.

Recommended progression:

```text
v2 scaffold
→ v3 real-data-correct
→ v3 shadow
→ prospective validation
→ selective per-outcome promotion
```

Possible later progression:

```text
regularized logistic regression
→ gradient boosting challenger
→ hierarchical/personalized models
→ advanced longitudinal models
```

Only move to deep learning if the data volume and validation results justify it.

---

# 26. Definition of "Real ML" for MindMap

MindMap has a real production ML system when:

- [ ] model trains on consented real user data
- [ ] labels correspond to actual intended prediction horizons
- [ ] missingness is handled correctly
- [ ] train/calibration/validation/test sets are leakage-safe
- [ ] artifacts are reproducible
- [ ] model/version/data lineage is stored
- [ ] learned model runs off-device
- [ ] inference runs in controlled infrastructure
- [ ] model operates in shadow mode first
- [ ] prospective outcomes are automatically reconciled
- [ ] calibration is monitored
- [ ] performance is measured by outcome
- [ ] weak outcomes remain rules-only
- [ ] drift is monitored
- [ ] user-facing outputs remain gated
- [ ] model can abstain
- [ ] rollback is possible
- [ ] raw journal content is excluded from forecasting
- [ ] consent/revocation policies are enforced
- [ ] no clinical claims are made without appropriate validation

---

# 27. Final Engineering Direction

```text
MindMap App Plane
─────────────────
Next.js
Capacitor
Supabase Auth
Supabase PostgreSQL
Supabase RLS

          │
          │ controlled structured data
          ▼

MindMap ML Plane
────────────────
Canonical Dataset Builder
Feature Pipeline
Training Pipeline
Evaluation Pipeline
Model Registry
Batch Inference
Shadow Evaluation
Monitoring

          │
          ▼

AWS
───
S3
KMS
SageMaker
ECR
ECS Fargate
EventBridge
Secrets Manager
CloudWatch

          │
          │ safe versioned outputs
          ▼

Supabase Predictions / Summaries
          │
          ▼

App UI
```

The highest priority is not cloud infrastructure yet.

The highest priority is:

```text
REAL-DATA CORRECTNESS
```

Start with:

```text
ml/real-data-correctness
```

Then:

```text
ml/real-training-pipeline
```

Then:

```text
ml/shadow-evaluation
```

Then:

```text
infra/aws-ml-plane
```

---

# 28. Instructions for Cursor / Claude Code

When implementing this roadmap:

1. Inspect the current repo before changing anything.
2. Reuse existing ML modules rather than creating parallel systems.
3. Treat `ml/` as the canonical Python ML package.
4. Preserve existing safety contracts.
5. Preserve existing synthetic tests.
6. Do not remove rule-based fallback.
7. Do not weaken abstention.
8. Do not use raw journal text for forecasting.
9. Do not use service-role credentials client-side.
10. Do not modify production schema casually.
11. Create explicit migrations when schema changes are required.
12. Add tests before changing label semantics.
13. Run the complete ML quality gate after every phase:

```bash
cd ml
uv run pytest
uv run ruff check .
uv run mypy
```

14. Run relevant evaluation commands after ML changes.
15. Update:
   - `MODEL_CARD.md`
   - `VALIDATION_PLAN.md`
   - `DATA_DICTIONARY.md`
   - `PRODUCTION_READINESS.md`
   when behavior materially changes.
16. Keep the model in `rules_only` or shadow mode until real-data gates pass.
17. Make every production change reversible.

---

# 29. First Task for Claude Code / Cursor

Begin with an audit and implementation plan for:

```text
ml/real-data-correctness
```

Specifically:

```text
1. inspect current feature + label pipeline
2. document current "next logged entry" behavior
3. design calendar-day label semantics
4. add tests demonstrating missing-calendar-day cases
5. refactor forward-label generation
6. implement explicit ML field allowlist
7. filter soft-deleted records
8. design training-consent filtering
9. build canonical daily dataset assembler
10. add dataset-quality report
11. run full ML test/lint/type gate
12. report changes before moving to training infrastructure
```

Do not start AWS work until this phase is green.

---

## Core Principle

> Accuracy, calibration, provenance, safety, and abstention matter more than model complexity.

MindMap should say:

```text
not enough data yet
```

before it emits a prediction the evidence does not support.
