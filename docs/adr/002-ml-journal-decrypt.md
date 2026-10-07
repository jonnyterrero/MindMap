# ADR-002 — Decrypt journals for the ML graph pipeline via a Vercel endpoint

- **Status:** Accepted (2026-10-07). Implemented on `feat/ml-journal-decrypt`.
- **Decision makers:** Jonny (product). Recorded so engineering can proceed without re-litigating.
- **Related:** [ADR-001](001-journal-encryption.md) (envelope encryption); `docs/session-handoff-2026-09-24.md` decision #3; ML graph pipeline (`ml/mindmap_ml/serving/`).

## Context

ADR-001 encrypts journal bodies with a per-user DEK wrapped by a master key held only in Vercel env (`JOURNAL_ENCRYPTION_MASTER_KEY`). After cutover, `mindmap_journal_entries.content` is `NULL` for encrypted rows and the plaintext lives only in `body_encrypted`.

The verified-mindmap graph pipeline runs as a **GitHub Actions cron** (`.github/workflows/ml-graph-cron.yml` → `python -m mindmap_ml.serving.graph_batch`). It reads journal text via `read_journal_entries()` using the Supabase **service-role key**, which Actions already holds. Today that function selects `content` and silently drops rows where `content IS NULL` — so **every entry written after encryption cutover is invisible to the graph pipeline**. Coverage quietly decays.

The constraint (handoff decision #3): the master key must **not** be added to GitHub Actions secrets. A leaked Actions log would then decrypt any stolen database backup offline, in perpetuity, with no way to revoke it short of re-encrypting every user's journals under a new key.

## Options considered

1. **Vercel decrypt endpoint, pulled by the batch at run time.** A service-to-service route decrypts inside the process that already holds the master key and returns plaintext over TLS. The batch holds a *shared bearer secret*, not the master key.
2. **Put the master key in GitHub Actions** and re-implement the envelope unwrap in Python. Rejected: this is exactly what decision #3 forbids; a leaked log decrypts offline backups forever and the key is not practically rotatable.
3. **Run the graph pipeline inside Vercel** to co-locate with the key. Rejected: the pipeline is heavy Python (pandas, the Anthropic grounder, a trained calibrator); porting it onto Vercel functions to move the key 30 lines is a large, fragile rewrite.

## Decision

**Option 1.** Decryption happens in Vercel; the batch pulls plaintext transiently.

- **Route:** `GET /api/internal/ml/journal-entries` (Node runtime, `force-dynamic`, never cached). Service-role client reads non-deleted rows, `decryptJournalRows()` decrypts encrypted bodies (legacy plaintext rows pass through), returns `{ count, entries: [{ id, user_id, entry_date, content }] }`. Never logs decrypted content.
- **Auth:** bearer `ML_JOURNAL_DECRYPT_SECRET`, constant-time compared, **secure by default** (rejects when unset). This secret is **not** the master key: it authorizes live calls only, and rotating it in Vercel + the Actions secret is instant and requires no re-encryption.
- **Batch:** `read_journal_entries(client, since)` fetches from the endpoint when `ML_JOURNAL_DECRYPT_URL` + `ML_JOURNAL_DECRYPT_SECRET` are set; otherwise it falls back to the direct table read (plaintext only, encrypted rows skipped) so local/dev is unaffected. Plaintext lives only in the runner's memory and is discarded when the job exits. `mindmap_graphs` keeps storing only the quoted spans it already stored — **no second long-lived plaintext copy**.
- **Watermark:** the batch pulls only entries with `updated_at >= now − ML_JOURNAL_LOOKBACK_DAYS` (default 7) to bound how much plaintext any single call surfaces. The existing `content_sha` skip still dedups unchanged entries, so the watermark never causes a recompute or drops a changed entry (7 days ≫ the daily cron interval). `--all` disables the watermark and pulls the full history.

## Consequences

- **Residual risk (accepted):** a leaked `ML_JOURNAL_DECRYPT_SECRET` lets someone call the endpoint and pull recently-updated decrypted journals *while it is reachable*. This is strictly better than the master key being in Actions — it is instantly rotatable, scoped to one endpoint, bounded by the lookback window, and useless against offline backups — but it is not zero. Future hardening: request-count logging/alerting, and tighter scoping (e.g. only un-graphed entries).
- **Backfill / recovery:** the first run after enabling encryption, or any recovery, must use `--all` (no watermark) so entries older than the window are (re)built. Routine daily runs stay incremental.
- **Failure mode:** if a row is encrypted but the endpoint's master key is missing/wrong, the route returns 500 and the batch aborts loudly rather than silently skipping entries.
- **Operations:** set `ML_JOURNAL_DECRYPT_SECRET` (same value) in Vercel and as a GitHub Actions secret, and `ML_JOURNAL_DECRYPT_URL` (the production route URL) as an Actions secret. The master key stays only in Vercel.

## Deferred

- `mindmap_voice_notes.transcript` is still plaintext (its own ADR + migration — the next queued item).
- DEK rotation (`JOURNAL_ENCRYPTION_MASTER_KEY_V2`) is unchanged by this ADR.
