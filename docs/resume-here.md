# Where we left off — 2026-09-04 (session 2)

Supersedes the morning note. The merge parade is **done**: all 11 branches
landed on `main` and are deployed to production via Vercel. Phase 1 items
1.2, 1.4 (code side), and 1.6 shipped the same day.

## Merged + deployed today

- All 11 PR branches (see git log `c3c8ab2..e7af84d`), including
  `chore/dep-security-updates` after its Vercel preview built clean
  (next 16.3.4, @supabase/ssr 0.12.5, supabase-js 2.115 — 45 → 2 prod vulns).
- Conflicts resolved against the analytics commits that landed on main
  in between (`.env.example`, `package.json`, lockfile rebuilt).
- pnpm 11 `allowBuilds`: `core-js` and `@sentry/cli` denied (both are
  postinstall-only; sentry-cli is source-map upload we don't use).

## Shipped this session (all verified: typecheck 0, build 0, tests 40/40)

| Item | What landed |
|---|---|
| **G7/G10 cleanup** | Deleted orphaned `getBodySensations`; untracked `legacy/`, `legacy branches/`, `cursor-claude-legacy/`, `sleep-stress/` (kept on disk, gitignored) |
| **1.4 Rate limits** | Migration **024** (`mindmap_ai_usage` + `rpc_consume_ai_quota`, applied to prod). Daily per-user quotas: chat 150, reflection 25, voice 40, reports 6. Wired into all 4 user-initiated AI paths; cron exempt. Chat surfaces the 429 copy. |
| **1.4 Sentry** | `@sentry/nextjs` server+edge+client, inert until `NEXT_PUBLIC_SENTRY_DSN` is set. No PII, no replay, no request bodies. |
| **1.6 Guardrails** | Insights stay "unknown" until 5 check-in days (progress copy instead of scores off one noisy day). Thumbs up/down feedback → migration **025** `mindmap_insight_feedback` (applied to prod). History sparkline wired via previously-orphaned `getInsightHistory` (G8). |
| **1.2 Notifications** | Reminders card in Settings → `mindmap_reminders` CRUD → Capacitor local notifications (daily or per-weekday), deterministic ids, native-launch re-sync via `NotificationsBootstrap`. Web shows "rings on your phone" copy. |
| **App Store prep** | `frontend/resources/` (1024 icon upscaled from 512 + dark splashes), `frontend/mobile/ios/PrivacyInfo.xcprivacy`, LocalNotifications config in `capacitor.config.ts`, and **`docs/app-store-checklist.md`** (nutrition-label answers, review-notes guidance, Mac build steps). |

## Only-you (dashboard/manual) items — the current blockers

1. **0.1 Key rotation** — Supabase + Anthropic + Vercel env + GH Actions secrets. Still open, still gates the ML cron.
2. **Leaked-password protection** — Supabase Dashboard → Auth → Passwords → enable "Prevent use of leaked passwords" (advisor still flags it; needs Pro plan).
3. **Sentry DSN** — create the Sentry project, set `NEXT_PUBLIC_SENTRY_DSN` in Vercel env. Code is already live and inert.
4. **Mac session** — `pnpm cap:add:ios` + assets + privacy manifest copy per `docs/app-store-checklist.md` §2; then TestFlight.

## Still open from the Beta Launch Plan (code work)

- **1.3** Real `deleteAccount` end-to-end test against prod-like cascade (do it in TestFlight per checklist §4).
- **1.5** Field-level journal encryption **decision** (see `docs/compliance-readiness.md` §164.312(a)(2)(iv)) — business call before code.
- **Phase 2** — beta invites, Playwright E2E for core flows, verified-mindmap cron enable (after key rotation), ADR-001.
- Advisor WARNs parked: `pg_net` in public schema; anon GraphQL discoverability on ~40 tables (RLS blocks rows; a blanket `revoke select from anon` needs care around `legal_documents`).
- `recharts` 2→3 major upgrade (closes the last 2 lodash vulns) — own PR with visual chart review.

## Compliance posture (unchanged)

`docs/compliance-readiness.md` still governs: HIPAA applies the day a BAA is
signed, SOC 2 the day an enterprise buyer asks. Biggest gap remains journal
plaintext (1.5). Today's Sentry/rate-limit/guardrail work all moves the
§164.312 and SOC 2 "monitoring" rows forward.

---

# Where we left off — 2026-09-19 (brief check-in)

Two weeks after session 2 (2026-09-04). Almost nothing to do here — this
session was ~15 minutes of orientation + one safety commit. Then we
compacted; deeper work resumes next time.

## What's happened since session 2

Between 2026-09-04 and 2026-09-19 the user shipped, without me, more of
Phase 1:

- `d633e3d` **1.3 real cascade deletion** — accepted as PR #3
  (`feat/code-backlog-1.3-deletion`, merged 2026-09-08). Section "Still
  open from the Beta Launch Plan (code work)" above lists 1.3 as needing
  a TestFlight-side end-to-end test; the code side is now on main.
- `df4fbe8` `fix(ml)`: hardcoded public Supabase URL so cron jobs survive
  a bad URL secret. PR #4 merged 2026-09-10.
- `aba3753` `fix(auth)`: let Vercel cron reach `/api/cron/generate-reports`
  without a session cookie.

## What I did this session (2026-09-19)

- **Committed `9affd6e`** — `.gitignore` now excludes
  `password manager.txt` (untracked at repo root; one broad `git add`
  from leaking) and `/MindMap-2/` (Claude Code scratch dir). Pattern is
  `password*manager*.txt` so a rename catches too.
- **Did not touch** the user's in-flight work:
  - `docs/adr/001-journal-encryption.md` (untracked — this is the 1.5
    decision doc)
  - `docs/app-store-checklist.md` (+3 lines uncommitted)
  - `frontend/MOBILE.md` (+3 lines uncommitted)
  - `frontend/AGENTS.md`, `frontend/CLAUDE.md` (new agent-instruction
    files)

## When we resume, the ranked next-work choice

Presented but not chosen (user compacted before selecting). One of:

1. **Help refine + implement ADR-001** (journal encryption). The user
   was drafting `docs/adr/001-journal-encryption.md` when this session
   started. Options in `docs/compliance-readiness.md` §164.312(a)(2)(iv):
   envelope encryption (server can decrypt; keeps AI reflection working),
   zero-knowledge (breaks AI reflection + crisis detection), or "disk-
   only, document it honestly." The AI-side implications are load-
   bearing — envelope is almost certainly the pragmatic call.
2. **`recharts` 2 → 3 upgrade** to close the last two prod vulns
   (`lodash` via recharts). Semver-major with breaking chart API
   changes; needs a visual review of dashboard/insights charts.
3. **Playwright E2E for core flows** — check-in → journal → insights →
   medications. Phase 2 gate; existing auth + legal specs pass so the
   infra is there.
4. **Close the two parked Supabase advisor WARNs** — `pg_net` in public
   schema and anon GraphQL discoverability on ~40 tables (RLS blocks
   rows; `legal_documents` needs anon read so this must be surgical).

Everything else in the "Still open" list from session 2 is unchanged.

