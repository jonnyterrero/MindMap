# Uncommitted local changes — 2026-09-24

Captured from the desktop working tree on `docs/session-handoff-2026-09-24`. These files were **not** committed. Apply them in the desktop session, then delete this capture if the real files are in git.

Nothing here is a secret. `frontend/.env.local` was not copied.

## 1. `docs/app-store-checklist.md` (modified)

Insert these three lines immediately after the heading `## 2. Mac-side build steps (cannot be done on Windows)` and before the bash fence:

```markdown
Parked 2026-09-09: no Mac available. Do not run `cap add ios` on Windows.
Native scaffolding in §1 stays; resume this section when a Mac exists.

```

## 2. `frontend/MOBILE.md` (modified)

Insert these three lines immediately after the heading `## One-time setup (on a Mac)` and before `Prerequisites:`:

```markdown
Parked 2026-09-09: no Mac available. Leave this section until then; do not
generate `ios/` on Windows. Config, resources, and PrivacyInfo already exist.

```

## 3. `frontend/CLAUDE.md` (untracked)

Entire file:

```markdown
@AGENTS.md
```

## 4. `frontend/AGENTS.md` (untracked)

Entire file. `next dev` rewrites this block; committing it keeps the tree clean.

```markdown
<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->
```

## 5. `docs/adr/001-journal-encryption.md` (untracked)

Accepted ADR. The status line below still says implementation has not started. That line is stale: the code shipped on `main` in `6fd1580` and `28fe3ca`. Update the status when you commit the real file.

```markdown
# ADR-001 — Envelope encryption for journal content

- **Status:** Accepted (2026-09-10). Implementation **not started**; next coding session.
- **Decision makers:** Jonny (product). Recorded so engineering can proceed without re-litigating.
- **Related:** `docs/compliance-readiness.md` §164.312(a)(2)(iv); launch audit item #5; plan item 1.5.

## Context

`mindmap_journal_entries.content` is stored as plaintext. Migration 002/008 already added `body_encrypted`, `encryption_key_id`, `encryption_algo` (`none` | `aes-256-gcm` | `xchacha20-poly1305`), `encrypted_at`, and an exclusivity check: either plaintext **or** ciphertext, never both. Nothing writes those columns.

Journal text is the highest-sensitivity field in the product. Server-side features that **must keep working** read it:

- Crisis keyword scan on save/sync
- Opt-in AI reflection
- Weekly AI reports
- ML graph batch (`read_journal_entries` quotes journal text)
- Data export
- Provider shares when `detail_level = 'full'`

Privacy copy currently says journals "support application-level encryption." That is ahead of the code and must be fixed in the implementation PR.

## Options considered

1. **Envelope encryption** — per-user data key, wrapped by a master key we hold. Server can decrypt. AI/ML/crisis/export/provider-full keep working.
2. **Zero-knowledge (client-side)** — key derived from the user password, never on the server. Server cannot decrypt. Those features break or must be rebuilt on-device.
3. **Disk-only** — keep Supabase AES-256 at rest; document that we do not do field-level encryption; stop claiming application-level encryption.

## Decision

**Envelope encryption (option 1). Keep all AI and server-side journal features.**

- Algorithm: **AES-256-GCM** (already allowed by the column check).
- Scope: **journal body only** (`content` → `body_encrypted`). Title, mood tags, dates, and other tables stay as they are.
- Keying: **per-user data encryption key (DEK)**, wrapped by a **master key** held in Vercel env (`JOURNAL_ENCRYPTION_MASTER_KEY`). DEK ciphertext stored outside the journal row (new `mindmap_journal_user_keys` table, or equivalent). `encryption_key_id` on the journal row points at that wrapped key.
- Decrypt on the server for every existing reader listed above. Do not send ciphertext to Anthropic; decrypt, call the model, do not log the body.
- Existing plaintext rows are **backfilled** (encrypt, null `content`, set `encryption_algo`). New writes never leave plaintext in `content`.
- Master key can later move to a real KMS (AWS KMS / GCP KMS / Vault) without changing the row format. Do not block the first PR on buying KMS.
- This is **Path A** in `docs/compliance-readiness.md`: a security control for a consumer app. It is **not** a HIPAA or SOC 2 claim.

## Consequences

**Positive**

- Stolen DB / backup snapshots of journal bodies are ciphertext.
- Schema already matches this design; we are wiring it, not inventing a new model.
- Product stays intact (companion, reflection, reports, mindmap graphs, crisis, export).

**Negative / accepted**

- The server (and therefore anyone with the master key + service role) can still read journals. That is the point of envelope vs zero-knowledge.
- Anthropic still sees plaintext **in the request** for opt-in AI features. Privacy policy must say so. Envelope does not hide journals from the model vendor.
- Python ML batch must share the same unwrap/decrypt path (or a small shared spec). Do not leave `graph_batch.py` reading `content` after backfill — it will be NULL.
- Key loss = journals unreadable. Master key lives in Vercel (and a documented offline backup). Rotation procedure is part of the implementation PR.
- `/privacy` encryption sentence must be rewritten in the same PR so copy matches reality.

**Explicitly out of scope (do not do in the encryption PR)**

- Claiming HIPAA compliance or SOC 2 in UI, marketing, or App Store text.
- Signing vendor BAAs (Supabase Team, Vercel Enterprise, Anthropic). Separate business track.
- Zero-knowledge rewrite.
- Encrypting check-in notes, chat transcripts, or AI analysis tables. Journal body only unless a later ADR expands scope.

## Compliance posture (not a legal opinion)

HIPAA does not automatically apply to a consumer self-tracking app with no covered-entity / business-associate contract. SOC 2 is an auditor attestation, not a law. Envelope encryption is one technical control. Counsel review of privacy/terms and FTC Health Breach Notification Rule coverage is a **user/legal** track, not this ADR.

Until counsel + BAAs (and for SOC 2, an auditor) exist: **do not claim HIPAA or SOC 2.**

## Implementation notes for the next session

Do not start until this ADR is the source of truth (it is, as of 2026-09-10).

1. Add `JOURNAL_ENCRYPTION_MASTER_KEY` to `frontend/.env.example` (placeholder only) and Vercel Production + Preview + Development after generating a 32-byte key.
2. New table for wrapped per-user DEKs + RLS (user cannot read another user's wrapped key; service role for cron/ML).
3. Server helper: `encryptJournalBody` / `decryptJournalBody` used by `journal/actions.ts`, `journal/sync-actions.ts`, `journal/voice-actions.ts`, export, report/AI paths, provider full-detail reads.
4. ML: decrypt in `serving/supabase_io.py` `read_journal_entries` (or decrypt before the pipeline). Same algo and key-id convention.
5. Backfill: one-shot script or migration-time job; idempotent; exclusivity constraint must hold after each row.
6. Tests: round-trip, exclusivity, "AI still receives plaintext," "DB row has null content," no plaintext in logs.
7. Fix `frontend/app/(legal)/privacy/page.tsx` security section.

Estimated effort: ~2 weeks as previously scoped. First slice can be helper + write path + backfill, then readers.

## Rollback

Set new writes back to `encryption_algo = 'none'` only if decrypt is proven broken in prod **and** we still have ciphertext + keys. Prefer fix-forward. A decrypt-and-restore-plaintext rollback is a deliberate data-classification downgrade and needs an explicit decision, not a silent revert.
```
