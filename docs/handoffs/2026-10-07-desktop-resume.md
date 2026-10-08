# Desktop resume — night of 2026-10-07

Read this first tomorrow. `origin/main` is `fd34eb5`. Repo: https://github.com/jonnyterrero/MindMap. Prod: https://getmindmapplus.app.

```bash
git fetch origin
git switch main
git pull
```

Do not build in `MindMap/MindMap-2`. That folder is gitignored scratch.

## Shipped tonight (already on main)

- `b0a87af` — Home is the hub. Returning sign-in, password reset, consent, and auth callbacks land on `/home`. Finishing onboarding still opens `/today` ("Start Day 1 Check-In"). Dashboard is in More. Companion button is hidden on the check-in page and sits above the phone dock. Complete check-in sticks above the dock.
- `f91a97d` + `693bba1` — ML graph decrypts journals inside Vercel (`GET /api/internal/ml/journal-entries`). The master key stays out of GitHub Actions. Middleware lets that path through; the route still checks its own bearer token.
- `51ba001` — ADR-003. Voice notes no longer write a plaintext `transcript`. The linked journal entry is the only copy. Migration `028` is in the repo.
- `fd34eb5` — Check-ins, streaks, meds, weather, body map, and new journal/therapy/med dates use the profile timezone (`America/New_York` if unset), not UTC. Evening logs in the US were landing on the next calendar day.

`ML_JOURNAL_DECRYPT_SECRET` is set in Vercel (all targets) and GitHub Actions. `ML_JOURNAL_DECRYPT_URL` is `https://getmindmapplus.app/api/internal/ml/journal-entries` and is set in GitHub Actions. An unauthenticated call returns the route's `{"error":"unauthorized"}`. A signed call that returns 200 has not been confirmed.

## Do this first (you, dashboards — no code)

Encryption is in the app and stays off until the master key exists. `JOURNAL_ENCRYPTION_MASTER_KEY` is still missing from `frontend/.env.local`.

1. Generate 32 random bytes. Store them in a password manager as `MindMap JOURNAL_ENCRYPTION_MASTER_KEY`. Do not paste the key into chat, commits, or GitHub Actions.
2. Put that same value in Vercel Production, Preview, and Development.
3. Apply migration `027` (`supabase/migrations/027_journal_encryption_user_keys.sql`) on production Supabase if it is not already there.
4. Create a journal entry. Expect `content` null, `encryption_algo = aes-256-gcm`, and `body_encrypted` present. Reload `/journal`. Settings export should still return plaintext.
5. Backfill older rows with `frontend/scripts/backfill-journal-encryption.ts` (dry-run, then `--apply`).
6. Apply migration `028` on production if it is not already there. It nulls `mindmap_voice_notes.transcript`. Production had 0 voice-note rows when ADR-003 was written.
7. After cutover, the first ML graph run must use `--all`. Later daily runs stay on the 7-day window. Do not write a second plaintext copy into `mindmap_graphs`.

Confirm these; the repo cannot see dashboard state:

- Plan **0.1** secret rotation (Supabase, Anthropic, Vercel, GitHub Actions). The 2026-09-10 scratch note said the ML cron was already green. The 2026-09-24 handoff still listed rotation as open.
- Sentry. The scratch note said `NEXT_PUBLIC_SENTRY_DSN` was on Vercel. The September handoff still listed creating the project.
- Leaked-password protection in Supabase Auth. Needs Pro. Leave it parked.

## Uncommitted on this machine — do not sweep them in with `git add .`

- `docs/adr/001-journal-encryption.md` — accepted ADR. The status line still says implementation has not started. That is stale: the code shipped in `6fd1580` and `28fe3ca`. Cutover has not started. Update the status line, then commit it.
- `frontend/AGENTS.md` and `frontend/CLAUDE.md` — Next.js agent stubs. Worth their own commit.
- `docs/app-store-checklist.md` and `frontend/MOBILE.md` — three lines each: parked 2026-09-09, no Mac, do not generate `ios/` on Windows.

## Code, when you want a coding session

In this order:

1. Commit the ADR-001 status fix and the two agent stubs above.
2. Playwright for check-in → journal → insights → medications. Specs can be written without credentials. A full local run needs `E2E_TEST_EMAIL` and `E2E_TEST_PASSWORD` in `frontend/.env.local`. CI stays skipped until `NEXT_PUBLIC_SUPABASE_*` repo variables exist.
3. Check-in under 90 seconds, Day-N progress, and a real Day-10 baseline report (plan **1.1**).
4. Onboarding welcome / next-steps and the medical disclaimer (plan **0.7**). No HIPAA or SOC 2 claims.
5. `recharts` 2 → 3, in its own PR, with a visual pass on the dashboard and insights charts.
6. Offline queue for check-ins and medication logs (plan **2.3**). The journal queue already exists.
7. Supabase advisor warnings: move `pg_net` out of `public`, and tighten anonymous GraphQL discovery. Keep anonymous read on `legal_documents`.
8. Later: drop `mindmap_voice_notes.transcript` after a bake period (ADR-003). DEK rotation via `JOURNAL_ENCRYPTION_MASTER_KEY_V2` needs no schema change.

## Leave alone

- iOS / TestFlight. No Mac. Resume from `docs/app-store-checklist.md` §2 on a Mac.
- Real ML training and AWS hosting (`ml/REAL_ML_TRAINING_AND_HOSTING_HANDOFF.md`).
- Wearable device connections, clinician PDF export, provider/B2B, SOC 2.
- Do not put `JOURNAL_ENCRYPTION_MASTER_KEY` in GitHub Actions.

## Stale notes

`docs/local-notes/2026-10-07-daily-loop-nav.md` and `docs/local-notes/2026-10-07-local-calendar-day.md` say those branches were not merged. Both landed on main tonight. `docs/session-handoff-2026-09-24.md` is an older branch and is behind this file.

Local branch `feat/journal-encryption-scaffold` is dangling. Its content is already on main. Delete it with `git branch -D feat/journal-encryption-scaffold` when you want the local list cleaned up. Several other local branches show `gone` on the remote; they are leftovers, not open work.
