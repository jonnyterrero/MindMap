# Session handoff — 2026-10-07

Resume point for the desktop. This session ran alongside a parallel Cursor
window (UI/UX + bug fixes), so `main` moved several times; everything below is
against `origin/main` @ **`fd34eb5`**. Builds on `docs/resume-here.md` and
`docs/session-handoff-2026-09-24.md`.

## Shipped this session (all merged to `main` + deploying/live)

| Commit | What |
|---|---|
| `5686217` | **Routine preset picker + Companion FAB.** Preset dropdown on /routines (`frontend/lib/routine-presets.ts`); global floating Companion button (`frontend/components/companion-fab.tsx`). Also fixed the optimistic temp-id bug in `routines-list.tsx`. |
| `f91a97d` | **ADR-002 — ML journal decrypt endpoint.** `GET /api/internal/ml/journal-entries` decrypts journals server-side so the GH Actions graph cron never holds the master key. Python `read_journal_entries(client, since)` fetches from it (fallback = direct read). 7-day incremental watermark; `--all` for full backfill. |
| `693bba1` | **Middleware fix for ADR-002.** The app middleware 401'd every `/api/**` without a Supabase session; added an `/api/internal/ml/*` exemption (mirrors `/api/cron/*`). Without it the cron always got 401. Found in prod verification. |
| `51ba001` | **ADR-003 — voice transcript minimization.** Stop writing the plaintext `mindmap_voice_notes.transcript` duplicate; the linked (encrypted) journal entry is the sole home. Migration **028 applied to prod** (0 rows; adds deprecation comment). |

Parallel Cursor commits also on main this session: `b0a87af` (land returning
users on Home; keep check-in tappable — fixes the Companion FAB overlapping the
check-in button), `fd34eb5` (store daily log on the user's calendar day).

## Operational state (ML decrypt — ADR-002)

- **Endpoint:** `https://getmindmapplus.app/api/internal/ml/journal-entries`
- **Secrets set by Jonny:** `ML_JOURNAL_DECRYPT_SECRET` in Vercel (all targets) + GitHub Actions; `ML_JOURNAL_DECRYPT_URL` in GitHub Actions. Secret is NOT the master key — rotatable in those two places, no re-encryption needed.
- **Verified in prod:** unauth call returns the route's `{"error":"unauthorized"}` (handler reached past middleware); 401 enforced.
- **STILL TO VERIFY (you, with the token):** the authenticated 200 path —
  ```bash
  curl -s -H "Authorization: Bearer <token>" https://getmindmapplus.app/api/internal/ml/journal-entries | head -c 200
  # expect {"count":N,"entries":[...]}
  ```
  The next ML graph-cron run also exercises it for real.
- Infra: Vercel project `mind-map` (`prj_z3CbFJmQOYdT3lExWCwzFSmSBkxr`, team `team_n7vQAxlQGjD9IcM7j8RECFzx`); Supabase `MindMap+_Backend` (`zunpccwjghwpiljwwjpv`).

## TODO — pick up here

### 1. Delete the dangling scaffold branch (30 sec, local)
Its content is on `main` as `6fd1580` + `28fe3ca`; safe to force-delete. Safety Net blocks `-D` from the agent, so run it yourself:
```bash
git branch -D feat/journal-encryption-scaffold
```

### 2. Playwright E2E (check-in → journal → insights → meds)
Not started — deliberately deferred. Two reasons, and a proposed split:
- **Shared working dir with Cursor** → do this work in an isolated `git worktree`, never the main checkout.
- **The target flows are exactly what Cursor is reworking** → specs written now would churn.
- **Plan:** (a) build the harness now in a worktree — `playwright.config.ts`, CI workflow (stays skipped until `NEXT_PUBLIC_SUPABASE_*` repo vars are set), an authed storage-state fixture, a login/landing smoke test; (b) write the flow specs after Cursor's UI changes settle.

### 3. ADR-001 journal-encryption cutover — NOT DONE (the big one)
Encryption is built and merged but **inert**: `JOURNAL_ENCRYPTION_MASTER_KEY` is not set in Vercel, so journals are still stored plaintext. Until this is done, the ML decrypt endpoint just returns plaintext (works fine, nothing to decrypt yet). Cutover steps live in `docs/session-handoff-2026-09-24.md`:
1. Generate the master key locally (PowerShell snippet in that doc) → save in password manager.
2. Set `JOURNAL_ENCRYPTION_MASTER_KEY` in Vercel (Prod+Preview+Dev).
3. Apply migration `027` to prod (may already be applied — verify).
4. Create a journal entry, confirm `content` null + `encryption_algo='aes-256-gcm'`.
5. Backfill: `frontend/scripts/backfill-journal-encryption.ts` (dry-run, then `--apply`).
- The ADR-001 draft (`docs/adr/001-journal-encryption.md`) is still an **uncommitted local file** — commit it (update its status line to Accepted/Implemented).

### 4. Remaining beta-gate items (from prior handoffs, unchanged)
- **0.1 Key rotation** — Supabase, Anthropic, Vercel env, GH Actions secrets. Still gates nothing new but overdue.
- Supabase **leaked-password protection** (Auth → Passwords; needs Pro).
- **Sentry DSN** — create project, set `NEXT_PUBLIC_SENTRY_DSN` in Vercel (SDK already deployed, inert).
- **recharts 2 → 3** upgrade (closes last lodash vulns) — needs visual chart review.
- **Supabase advisor WARNs** — `pg_net` in public schema; anon GraphQL discoverability (keep `legal_documents` anon-readable). Backend-only, won't collide with Cursor — good candidate to do next.
- **Deferred from ADR-003:** drop the `mindmap_voice_notes.transcript` column in a later migration after a bake period.
- **TestFlight** (Mac only): `pnpm cap:add:ios`, assets, privacy manifest — parked until a Mac exists.
- 0.7 onboarding/disclaimer copy; 1.1 check-in <90s + Day-10 report; 2.3 offline queue for check-ins/meds.

## Coordination notes (important)
- **This repo's working tree is shared with the Cursor window.** It got switched between branches under the agent mid-session. Always `git fetch` before branching/merging; use a `git worktree` for any work so the two don't fight. Merges this session used remote fast-forward pushes (`git push origin <branch>:main`) to avoid disturbing Cursor's checkout.
- **Workflow rule:** every change → its own branch → pushed → reviewed on Vercel preview → merged only on approval → stale branch deleted.

---
_Written by Claude (Opus 4.8) for Jonny, 2026-10-07._
