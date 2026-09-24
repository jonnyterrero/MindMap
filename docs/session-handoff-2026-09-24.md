# Session handoff — 2026-09-24

Resume this session from GitHub on the desktop or in the cloud. This file is the source of truth for the 2026-09-22 Cursor session plus the desktop state at handoff time. It does not replace `docs/resume-here.md` (Claude, through session 4 on 2026-09-22).

## Where the code is

- Repo: https://github.com/jonnyterrero/MindMap
- This branch: `docs/session-handoff-2026-09-24`
- `origin/main` at handoff: `1d3c4fc` (session 4 check-in; ADR-001 code already on main)
- Desktop git root: `C:\Users\JTerr\OneDrive\Programming Projects\Mindmap+\MindMap`
- Do not build in `MindMap/MindMap-2`. That directory is a scratch folder and is gitignored.

This branch is `origin/main` plus two local commits that were **not** on `origin/main` when the handoff was written, plus this file:

- `5e738b8` `fix(privacy): enforce analytics opt-in` — plan item **1.7**. Vercel Analytics and Speed Insights load only after consent.
- `697a74c` `chore(security): patch production dependencies` — dependency patch on this branch. The `recharts` 2 → 3 upgrade is still a separate PR and still needs a visual chart review.

Left uncommitted on the desktop on purpose (do not assume they are on this branch):

- `docs/app-store-checklist.md` and `frontend/MOBILE.md` — "Parked 2026-09-09: no Mac available"
- `docs/adr/001-journal-encryption.md` — accepted ADR; status line still says implementation has not started. Update that line and commit it.
- `frontend/AGENTS.md`, `frontend/CLAUDE.md` — Next.js agent stubs worth committing

## Decisions made this session

1. OpenSSL is not required. `JOURNAL_ENCRYPTION_MASTER_KEY` is 32 random bytes, hex (64 chars) or base64. The app reads it from `frontend/.env.local` (gitignored). `frontend/.env.example` already has an empty placeholder.
2. As of this handoff the key line is **not** in `frontend/.env.local`. Cutover has not started.
3. Do **not** put `JOURNAL_ENCRYPTION_MASTER_KEY` in GitHub Actions. The ML cron already has `SUPABASE_SERVICE_ROLE_KEY`. Adding the master key there means a leaked Actions log can unwrap every journal. Leave encrypted rows skipped until decrypt runs inside the Vercel process that already holds the key, then discard the plaintext.
4. `mindmap_graphs` stores quoted journal text. A later ML decrypt path must not write a second long-lived plaintext copy.
5. Do not paste the master key into chat, commits, or `.env.example`.

## Next action — generate the key locally

Run this in PowerShell. It writes `frontend/.env.local` and copies the key to the clipboard. It does not print the key.

```powershell
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$hex = -join ($bytes | ForEach-Object { $_.ToString('x2') })
Add-Content -Path "C:\Users\JTerr\OneDrive\Programming Projects\Mindmap+\MindMap\frontend\.env.local" -Value "`nJOURNAL_ENCRYPTION_MASTER_KEY=$hex"
Set-Clipboard -Value $hex
"Key written to frontend/.env.local and copied to the clipboard."
```

Then:

1. Paste the clipboard into a password manager entry named `MindMap JOURNAL_ENCRYPTION_MASTER_KEY`. Clear the clipboard after that. Losing this value after cutover makes journals unreadable.
2. Confirm without printing the secret: length 64 and hex-only.
3. Put the **same** value in Vercel Production, Preview, and Development before any journal write against the production database. A key that exists only in `.env.local` will encrypt a production row that production cannot decrypt.
4. Restart `pnpm dev` after editing `.env.local`.

## Cutover checklist (after the key exists in Vercel)

Do these in order. Detail lives in `docs/resume-here.md` under "2026-09-22 (session 4)".

1. Apply migration `027` (`supabase/migrations/027_journal_encryption_user_keys.sql`) on production Supabase.
2. Set `JOURNAL_ENCRYPTION_MASTER_KEY` in Vercel Production, Preview, and Development.
3. Create a journal entry and confirm `content` is null, `encryption_algo = aes-256-gcm`, and `body_encrypted` is present.
4. Reload `/journal` and confirm Settings "Export my data" returns plaintext `content`.
5. Backfill historical rows with `frontend/scripts/backfill-journal-encryption.ts` (dry-run, then `--apply`).

## Still open, in order

Manual, dashboard only:

- **0.1** Rotate Supabase, Anthropic, Vercel env, and GitHub Actions secrets. This still gates the ML cron.
- Supabase leaked-password protection (Auth → Passwords; needs Pro).
- Create a Sentry project and set `NEXT_PUBLIC_SENTRY_DSN` in Vercel. The SDK is already deployed and inert.
- On a Mac only: `pnpm cap:add:ios`, assets, privacy manifest, TestFlight (`docs/app-store-checklist.md` §2). Parked until a Mac exists.

Code:

- ML decrypt of journals **inside Vercel**, not in GitHub Actions. Until then, graphs skip encrypted entries. Do not persist a second plaintext copy in `mindmap_graphs`.
- ADR plus migration for `mindmap_voice_notes.transcript` (same text is still plaintext).
- Data-encryption-key rotation later via `JOURNAL_ENCRYPTION_MASTER_KEY_V2`. No schema change.
- Delete the local dangling branch `feat/journal-encryption-scaffold` when back on the desktop.
- TestFlight end-to-end pass of account deletion (`d633e3d` is already on main).
- Playwright E2E for check-in → journal → insights → medications. CI stays skipped until `NEXT_PUBLIC_SUPABASE_*` repo variables are set.
- `recharts` 2 → 3, with a visual review of dashboard and insights charts.
- Supabase advisor warnings: `pg_net` in the public schema, and anonymous GraphQL discoverability. Keep `legal_documents` anonymously readable.
- **0.7** Onboarding funnel and disclaimer copy pass.
- **1.1** Check-in under 90 seconds, 10-day baseline UX, and a real Day-10 report.
- **2.3** Offline queue for check-ins and medication logs. Journal queue already exists.

**1.7 analytics opt-in is done on this branch** (`5e738b8`), not on `origin/main`.

Beta, after the gate above: invite 10–20 users, first real 10-day baseline reports, interviews plus the FGCU one-pager, and expand the ML gold set toward ~200 dual-annotated entries. Learned forecasting stays parked until about 30–50 users × 6–8 weeks.

## How to resume

```bash
git fetch origin
git switch docs/session-handoff-2026-09-24
```

Read this file, then `docs/resume-here.md` session 4. First concrete task is the key generation block above, then the cutover checklist. Do not copy the master key into GitHub Actions.
