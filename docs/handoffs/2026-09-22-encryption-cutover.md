# Session Handoff — 2026-09-22 (ADR-001: Journal Envelope Encryption)

**Written:** 2026-09-24 for cloud + desktop resumption
**Session date:** 2026-09-22
**Session identifier:** Session 4 (following the pattern in [docs/resume-here.md](../resume-here.md))
**State of repo at handoff:** [main @ `1d3c4fc`](https://github.com/jonnyterrero/MindMap/commit/1d3c4fc), clean sync between local + `origin/main`
**Purpose of this doc:** Self-contained brief so a fresh cloud session (or the desktop, whichever picks up next) can execute the cutover and its follow-ups without re-deriving anything.

> **Read alongside:** [`docs/session-handoff-2026-09-24.md`](../session-handoff-2026-09-24.md) — the parallel handoff written from the cloud session's perspective. That doc is authoritative for **decisions made about key handling** (in particular: do NOT put `JOURNAL_ENCRYPTION_MASTER_KEY` in the GitHub Actions env — leak risk from any Actions log). This doc is longer and covers the ADR-001 implementation history + cutover mechanics; the cloud handoff has fresher operational context.

---

## TL;DR

- **All code for ADR-001 (journal envelope encryption) is on `main`.**
- **Encryption is OFF in production** because `JOURNAL_ENCRYPTION_MASTER_KEY` is not set in Vercel. That is deliberate — every wired site guards on `isEncryptionEnabled()`, so the merged code behaves identically to the pre-merge behavior until the env var flips.
- **The cutover is a 6-step manual playbook** (below). It happens in the cloud (Supabase dashboard, Vercel dashboard, one node script run). No further coding is required for the switch-on.
- **Three deferred follow-ups** are documented at the end. The ML graph pipeline is the most consequential — it silently skips encrypted rows after cutover, so entries logged post-cutover won't feed the mindmap graph until an ML-side decrypt PR ships.

---

## What shipped this session

### Commit `6fd1580` — encryption scaffold

Storage + primitives, zero behavior change.

| File | What it does |
|---|---|
| [supabase/migrations/027_journal_encryption_user_keys.sql](../../supabase/migrations/027_journal_encryption_user_keys.sql) | Creates `mindmap_journal_user_keys` table with per-user wrapped Data Encryption Keys (DEKs). RLS: user can `SELECT` their own wrapped rows, writes are service-role only. Partial unique index `uq_mindmap_journal_user_keys_active (user_id) WHERE is_active = true` enforces one active DEK per user; historical rows are preserved so entries under old key ids still decrypt after rotation. |
| [frontend/lib/journal-crypto.ts](../../frontend/lib/journal-crypto.ts) (primitives half) | Envelope AES-256-GCM primitives. Master key from `JOURNAL_ENCRYPTION_MASTER_KEY` (hex-64 or base64-44), versioned via a `wrap_master_key_version` field on every wrap row so a later rotation adds `_V2` without touching the schema. Body blob is `iv(12) ‖ tag(16) ‖ ciphertext(N)` packed into the existing `body_encrypted bytea` column. |
| [frontend/tests/journal-crypto.test.ts](../../frontend/tests/journal-crypto.test.ts) | 15 unit tests: hex/base64 master-key parsing, wrong-length rejection, secret never appearing in error messages, wrap/unwrap round-trip, wrong master key trips GCM auth, body round-trip including empty string + unicode, IV randomness, minimum-blob guard (exactly `IV+tag = 28` bytes), bit-flip caught, wrong DEK caught, wrong-length DEK rejected before the blob check. Ran via `pnpm test` (Node's built-in `node --test`). |
| [frontend/.env.example](../../frontend/.env.example) | `JOURNAL_ENCRYPTION_MASTER_KEY` placeholder with `openssl rand -hex 32` generation instructions. |

### Commit `28fe3ca` — end-to-end wire-up

Every read/write path guarded by `isEncryptionEnabled()`; unset key ⇒ legacy plaintext path.

| File | What it does |
|---|---|
| [frontend/lib/journal-crypto.ts](../../frontend/lib/journal-crypto.ts) (DB-aware half) | Adds `isEncryptionEnabled`, `getOrCreateActiveUserDek` (with race-safe retry via the partial unique index), `getUserDekById`, `encryptJournalBodyForUser` (returns the DB patch callers spread into `insert`/`update` with `content: null`), and `decryptJournalRow` / `decryptJournalRows` (batch-decrypt with per-key DEK cache so 50 rows under one key = one unwrap). |
| [frontend/app/(app)/journal/actions.ts](../../frontend/app/(app)/journal/actions.ts) | `createJournalEntry` encrypts on insert (crisis detection uses `payload.content` — the plaintext we already have — not a re-read of the row). `updateJournalEntry` re-encrypts only when `content` is in the patch; title/mood/privacy edits pass through untouched, preserving whatever encrypt state the row already has. `reflectOnJournalEntry` selects the encryption columns and decrypts before shipping to Anthropic (envelope encryption protects the DB, not the AI request path — `/privacy` copy updated to say so). `getJournalEntries` batch-decrypts. |
| [frontend/app/(app)/journal/sync-actions.ts](../../frontend/app/(app)/journal/sync-actions.ts) | Offline-flush encrypts each queued payload; crisis detection walks `payloads[i].content` aligned to returned ids (row-back reads would be null under encryption). |
| [frontend/app/(app)/journal/voice-actions.ts](../../frontend/app/(app)/journal/voice-actions.ts) | Voice-note journal insert encrypts. **Caveat:** `mindmap_voice_notes.transcript` still stores plaintext — out of scope for ADR-001. Same text stored twice, once encrypted, once plaintext. See follow-up #2. |
| [frontend/app/api/export/route.ts](../../frontend/app/api/export/route.ts) | JSON + CSV export decrypts journal rows so downloads contain plaintext the user can read; ciphertext columns are nulled in the export. |
| [frontend/app/(app)/settings/data-privacy-actions.ts](../../frontend/app/(app)/settings/data-privacy-actions.ts) | `exportUserData` (the "Export my data" bundle) does the same decrypt + null-ciphertext pass. |
| [frontend/app/(legal)/privacy/page.tsx](../../frontend/app/(legal)/privacy/page.tsx) | Section 9 rewritten to describe what actually ships: AES-256-GCM envelope, per-user DEK, master key on the server, protects DB backups only, does not hide content from AI providers for opted-in features. |
| [frontend/scripts/backfill-journal-encryption.ts](../../frontend/scripts/backfill-journal-encryption.ts) | Backfill CLI. Batched (100 rows/iteration), idempotent (only touches rows still at `encryption_algo = 'none'`), dry-run by default, `--apply` to write. Never logs plaintext bodies. |

### Commit `1d3c4fc` — session 4 note

Appended session 4 log to [docs/resume-here.md](../resume-here.md) with the cutover playbook. This handoff doc supersedes the terser version there.

---

## Cutover playbook (do in the cloud, in this order)

Nothing below requires local machine access. All commands are either dashboard clicks or run in a cloud shell / Vercel Postgres console with the same env vars set.

### Step 1 — Apply migration 027 to production

Two options; either works:

- **Supabase MCP** (if the session has it authenticated): `apply_migration` with the contents of [supabase/migrations/027_journal_encryption_user_keys.sql](../../supabase/migrations/027_journal_encryption_user_keys.sql).
- **Supabase Dashboard**: SQL Editor → paste the file's contents → Run.

Idempotent-ish — the table uses `CREATE TABLE IF NOT EXISTS` and `CREATE POLICY IF NOT EXISTS`. Safe to re-run. New table, no data risk.

**Verify:**
```sql
SELECT COUNT(*) FROM public.mindmap_journal_user_keys;                -- expect 0 initially
SELECT policyname, cmd FROM pg_policies WHERE tablename = 'mindmap_journal_user_keys';
-- expect one row: journal_user_keys_owner_read, SELECT
```

### Step 2 — Generate a master key

```bash
openssl rand -hex 32
```

**Save this in a durable location** (password manager, offline vault). Losing it means every encrypted journal is unrecoverable forever — there is no key-recovery mechanism by design.

### Step 3 — Set the env var in Vercel

Vercel dashboard → Project **mind-map** → Settings → Environment Variables. Add:

- **Name:** `JOURNAL_ENCRYPTION_MASTER_KEY`
- **Value:** the hex string from step 2
- **Environments:** Production ✓ Preview ✓ Development ✓

Vercel triggers a redeploy on env-var change. Wait for the deploy to complete before step 4.

### Step 4 — Verify encrypt-on-write

Log in on `getmindmapplus.app` (or a preview URL that also has the env var). Create a journal entry with some distinctive text.

In Supabase SQL Editor:

```sql
SELECT
  id,
  content,                                        -- should be NULL
  encryption_algo,                                -- should be 'aes-256-gcm'
  encryption_key_id,                              -- should be 'k_' + 32 hex chars
  length(body_encrypted) AS ciphertext_bytes     -- should be >= 28 (IV + tag min)
FROM public.mindmap_journal_entries
ORDER BY created_at DESC
LIMIT 1;
```

If `content` is still populated and `encryption_algo = 'none'`, the env var didn't reach the deployment — check Vercel logs for the request. If `content = NULL` and the other three fields are set correctly, encryption is working.

### Step 5 — Verify decrypt-on-read

Two checks:

1. Reload `/journal` in the browser. The entry from step 4 should render with its original text (not gibberish, not blank).
2. `/settings` → **Export my data** → download the JSON bundle. Open it. The `mindmap_journal_entries` array should have `content` = plaintext and `body_encrypted` / `encryption_algo` / `encryption_key_id` / `encrypted_at` all null. That matches how the export helper strips ciphertext columns for GDPR readability.

Only proceed to step 6 after both round-trips work.

### Step 6 — Backfill historical rows

Only after step 5 confirms the read path decrypts. From any shell with the same `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and `JOURNAL_ENCRYPTION_MASTER_KEY` env set (Vercel CLI shell, cloud shell, or a re-imaged desktop):

```bash
cd frontend
# 1. Dry-run first — scans, logs counts, writes nothing.
node --import ./scripts/ts-resolve-hook.mjs scripts/backfill-journal-encryption.ts

# 2. Real run — encrypts + updates in batches of 100.
node --import ./scripts/ts-resolve-hook.mjs scripts/backfill-journal-encryption.ts --apply
```

Both runs print `scanned=… encrypted=… skipped=… failed=…`. Re-run `--apply` any time — the query filter is `WHERE encryption_algo = 'none'`, so already-encrypted rows drop out naturally. A mid-run crash resumes cleanly on the next invocation.

Verify:

```sql
SELECT encryption_algo, COUNT(*)
FROM public.mindmap_journal_entries
WHERE deleted_at IS NULL
GROUP BY encryption_algo;
-- expect only aes-256-gcm rows after a successful --apply run
```

---

## Rollback plan (if cutover goes wrong)

**Before step 6 (no historical rows encrypted yet):**

1. Vercel dashboard → delete `JOURNAL_ENCRYPTION_MASTER_KEY`. Wait for redeploy.
2. New writes revert to plaintext.
3. Reads: the handful of rows written under encryption during the botched window still have `encryption_algo = 'aes-256-gcm'` and are unreadable without the master key. Options:
   - (a) Restore the master key and use it read-only; those rows decrypt again.
   - (b) `UPDATE public.mindmap_journal_entries SET content = '[encrypted content lost]', body_encrypted = NULL, encryption_key_id = NULL, encryption_algo = 'none', encrypted_at = NULL WHERE encryption_algo = 'aes-256-gcm';` — deliberate data loss for a bad key, only if the key is truly gone.

**After step 6 (backfill ran):** rollback means keeping the key. There is no "decrypt everything back to plaintext" script (would be a data-classification downgrade — deliberately not shipped). If a full-decrypt rollback is genuinely needed, that's a small script that mirrors the backfill in reverse — but the strong recommendation is fix-forward.

---

## Deferred follow-ups (NOT done this session)

Each has a clear next step so a fresh session doesn't re-derive it.

### 1. ML graph pipeline decrypt

**File:** [ml/mindmap_ml/serving/supabase_io.py](../../ml/mindmap_ml/serving/supabase_io.py) around line 56 (`read_journal_entries`).

**Current behavior:** the function already filters out rows with null `content` (line 74). So after cutover, encrypted entries silently drop from the graph pipeline. **Not a crash, but the mindmap-graph feature loses coverage of every entry written after cutover** until this is fixed.

**How to fix — DO NOT put the master key in GitHub Actions.** Decision from the cloud handoff (see [session-handoff-2026-09-24.md](../session-handoff-2026-09-24.md) §Decisions): adding `JOURNAL_ENCRYPTION_MASTER_KEY` to the ML cron's GitHub Actions env means any leaked Actions log can unwrap every user's journal. The correct architecture is:

1. Decrypt inside the **Vercel process** that already holds the master key (a new server-side endpoint the ML batch can call, or move the graph batch trigger into a Vercel cron so it runs in-process).
2. The Vercel endpoint returns the plaintext body to the ML batch **only for the duration of the graph build** — the batch must not persist the plaintext anywhere.
3. `mindmap_graphs` currently stores quoted journal text; the ML decrypt path must not write a **second** long-lived plaintext copy — either scrub the quoted spans from graph payloads or store span offsets against the encrypted row so the plaintext can be re-fetched on demand.

**Not to do:** hand the master key to GitHub Actions, mount it into the batch's env, or copy plaintext bodies into any table other than the one already handling them.

**Estimated effort:** ~1-2 days (larger than "just add crypto to Python" because it needs the Vercel endpoint + a decision about `mindmap_graphs` quoted-text retention).

### 2. `mindmap_voice_notes.transcript` plaintext

**File:** [frontend/app/(app)/journal/voice-actions.ts:88](../../frontend/app/(app)/journal/voice-actions.ts#L88) writes `transcript: text` unencrypted. The same text is also written encrypted into `mindmap_journal_entries` a few lines above.

**Fix:** own ADR (call it ADR-002). Options mirror ADR-001 (envelope, zero-knowledge, or "don't"). Needs a new `transcript_encrypted bytea` + `transcript_encryption_algo text` on `mindmap_voice_notes`, plus an exclusivity CHECK, plus the same encrypt/decrypt path.

**Estimated effort:** ~2-3 days (smaller than ADR-001 because the DEK infrastructure already exists).

### 3. DEK rotation CLI

**Not urgent, but needed before a real key-compromise event.**

**Design:** add `JOURNAL_ENCRYPTION_MASTER_KEY_V2` to Vercel, run a script that (a) reads all currently-active DEKs, (b) re-wraps them under `_V2` with `wrap_master_key_version = 'v2'`, (c) marks V1 wrap rows as `is_active = false` and inserts the V2 wraps as `is_active = true`, (d) optionally re-encrypts journal bodies under the new active DEK (a per-user re-wrap of DEKs already lets old journal rows decrypt against the V1-wrapped DEK; re-encrypting the bodies too provides forward secrecy at the body-key level).

**No schema changes needed** — the `wrap_master_key_version` column already exists on `mindmap_journal_user_keys`.

**Estimated effort:** ~half a day.

### 4. Dangling local scaffold branch

Only relevant when the Windows desktop is back in use.

```bash
git branch -D feat/journal-encryption-scaffold
```

Content is on `main` under `6fd1580` + `28fe3ca` (verified). Git refused `-d` because the rebase changed hashes and the old commits are no longer strict ancestors of main. Remote scaffold branch already deleted.

---

## User's local in-flight work — NOT touched

The following are on the Windows desktop and were deliberately left alone this session (matches the session-3 pattern):

- `docs/app-store-checklist.md` (+3 uncommitted lines, "Parked 2026-09-09: no Mac available")
- `frontend/MOBILE.md` (+3 uncommitted lines, same parking note)
- `docs/adr/001-journal-encryption.md` (the ADR draft — status "Accepted 2026-09-10"; implementation is what shipped this session)
- `frontend/AGENTS.md` (untracked, auto-generated by `next dev`; content warns "This is NOT the Next.js you know" — worth committing before more agent-driven work)
- `frontend/CLAUDE.md` (untracked, `@AGENTS.md` re-export for Claude Code)

Zero overlap with anything committed this session. Safe for the desktop to pick these up whenever.

---

## Repository state summary

**Branch:** `main` @ [`1d3c4fc`](https://github.com/jonnyterrero/MindMap/commit/1d3c4fc)
**Origin:** in sync.
**Remote branches still open:** none from this session. Older merged branches from previous sessions (`chore/*`, `feat/edit-ux`, `feat/loading-states`, `feat/phase-0-ux`, `ci/ml-gate`, `docs/compliance-readiness`) still exist on the remote and are the user's to archive-tag + delete on their own cadence.

**Verification commands** (run any time to confirm the code compiles + tests pass):

```bash
cd frontend
pnpm typecheck                                                       # expect exit 0
pnpm build                                                           # expect exit 0
pnpm test                                                            # expect 55 pass, 1 pre-existing skipped
```

---

## Session context — what came before

Full history in [docs/resume-here.md](../resume-here.md). Short version:

- **Session 1** (2026-08-06): PR 1-5 batch. Shipped stub-API removal, RLS policy fixes (migration 023, applied live), env cleanup, ML CI gate, loading skeletons.
- **Session 2** (2026-09-04): PR 6-11 batch. Fake-deletion removal, backfill of 5 out-of-repo migrations, dep security updates (Next 16.3.4, Supabase SSR 0.12.5), edit UX, wearables "coming soon", compliance-readiness doc.
- Between session 2 and session 3, the user shipped Phase 1 items solo: real cascade deletion (PR #3), ml cron URL fix (PR #4), Vercel cron auth fix, local notifications (Phase 1.2), insight guardrails + feedback + trend (Phase 1.6 + G8), AI rate limits + Sentry (Phase 1.4), App Store prep, ML real-training + hosting handoff.
- **Session 3** (2026-09-19): 15-minute orientation, one safety commit (`9affd6e`: gitignore `MindMap-2/` scratch and `password* manager.txt`).
- **Session 4** (2026-09-22): THIS SESSION — ADR-001 both slices to `main`.

---

## When the cutover is done

Update `docs/resume-here.md` with a session-5 note listing the actual dates the switch-on happened, the size of the backfill, and any surprises. Move ADR-001 status from "Accepted, Implementation shipped 2026-09-22" to "Accepted, Live in prod 2026-09-XX (env: `JOURNAL_ENCRYPTION_MASTER_KEY` set; backfill: N rows)". Move `docs/compliance-readiness.md` §164.312(a)(2)(iv) from 🔴 to 🟢 (or 🟡 depending on how much of the follow-ups above are done).
