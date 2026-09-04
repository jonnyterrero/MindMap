# Where we left off — 2026-09-04

Paused mid-session, no cleanup owed. Everything below is on `main` at
`6544637` untouched; work landed as **11 pushed branches**, zero opened
PRs. Merge each on your own timing.

## What's live in production RIGHT NOW

Only one thing: **DB migration 023** (`rls_policy_fixes`) plus its follow-up
`rls_policy_fixes_revoke_anon` — applied via the Supabase MCP during the
session because you approved a dry-run-then-apply. Closes:

- Audit-log impersonation (`security_audit_events` INSERT policy)
- Provider RPC leak (`rpc_provider_get_entries` + `rpc_provider_get_insights` returning `SELECT *`)
- Geolocation leak (`profiles_provider_read` exposing `weather_lat`/`weather_lon`)

No frontend is deployed yet. That's harmless today because prod has **0
provider accounts and 0 active grants** — nothing hits the affected path.
The provider-page frontend that consumes the new RPC only merges when
`fix/rls-policy-bugs` reaches main.

## Branches on GitHub, in the order I'd merge them

Order is by risk × value; do PR 6 → PR 4 → PR 5 → PR 3 → PR 9 → PR 1 → PR 2
first, then the bigger ones. Each is verified independently: typecheck 0,
build 0, 39 unit tests 0.

| # | Branch | What it does | Risk |
|---|---|---|---|
| 1 | `fix/security-hygiene` | Delete the stub `/api/v1/*` + `/api/auth` + `lib/api-auth.ts`; unblock `/auth/callback`; ignore retired snapshot | Low |
| 2 | `fix/rls-policy-bugs` | **Frontend catch-up** for the migration already applied to prod. Merges the `rpc_provider_get_patient_profiles` call site + migration file itself | Low (DB is already there) |
| 3 | `chore/env-naming-cleanup` | `.env.example` gains `ANTHROPIC_API_KEY` + `CRON_SECRET`; package rename; SW dedupe; middleware matcher fix so `/sw.js` and `/manifest.json` load signed out | Low |
| 4 | `ci/ml-gate` | New CI job: `uv run pytest && ruff check && mypy` on the `ml/` package | Low |
| 5 | `feat/loading-states` | 11 per-route `loading.tsx` skeletons + shared primitives + custom `not-found.tsx` | Low |
| 6 | `fix/kill-fake-deletion` | Remove the fake "Request Data Deletion" card from settings (kept the real one in `data-privacy.tsx`) | Low, GDPR-critical |
| 7 | `chore/dep-security-updates` | `next 16.0.10 → 16.3.4`, `@supabase/ssr 0.8 → 0.12.5`, `@supabase/supabase-js 2.98 → 2.115`, `autoprefixer` + `postcss` bumps. **45 → 2 prod vulns, all Next.js middleware bypasses closed** | **Medium** — verify Vercel preview before merge |
| 8 | `feat/edit-ux` | Journal / medication / goal-progress inline editing (wires 3 dead server actions to real UI) | Low |
| 9 | `chore/backfill-out-of-repo-migrations` | 5 missing migrations recovered from prod (`011a-d`, `022a`) so repo == prod. All idempotent | Low (already in prod) |
| 10 | `feat/phase-0-ux` | Wearables "coming soon"; signup + consent link to Terms / Privacy / Medical / AI Disclosure | Low, copy-only |
| 11 | `docs/compliance-readiness` | New `docs/compliance-readiness.md` — HIPAA/SOC 2 inventory, no code | None |

Nothing to force-push, nothing to rebase. All 11 branch off `main` @ `6544637`
except PR 2 → 1 and PR 11 → main (they only share `main`).

## Two branches that share `middleware.ts`

Trial-merged during the session on `verify/all-prs` (local, not pushed) —
merged cleanly, no conflict:
- PR 1 adds `/auth/callback` to `PUBLIC_ROUTES`.
- PR 3 adds `sw.js` + `manifest.json` to the matcher exclusion.

Git handled it because the edits are in different regions. Verified against
a running prod server: `/today /settings /provider /dashboard /insights`
still 307→login; `/sw.js /manifest.json` return 200 signed out.

## Merge-order gotcha to remember

PR 7 (`chore/dep-security-updates`) is the one worth an extra look. It ran
clean end-to-end for me (typecheck, build, all 39 tests, runtime auth semantics
unchanged on port 3415), but Next 16.0.10 → 16.3.4 crosses minor versions and
your CI has type/lint errors ignored (`next.config.mjs` sets
`ignoreBuildErrors: true`), so **Vercel preview is the real gate**. Push to
its own preview branch first; if it deploys clean, merge.

The intentionally-deferred residual is 2 vulns: `lodash` via `recharts`,
neither directly exploitable (recharts doesn't use `_.template` or
user-controlled `_.unset` paths). Both close by upgrading `recharts` 2→3,
which is a semver-major with breaking chart API changes and deserves its
own PR with a visual review of the dashboard/insights charts. Left as a
follow-up.

## Still open from the Beta Launch Plan

None of these have branches yet; each is Phase 1 in the plan:

- **0.1 Key rotation** — Supabase + Anthropic + Vercel env + GitHub Actions secrets. Only you can do this in the dashboards; I cannot.
- **1.2 Local notifications** via `@capacitor/local-notifications` (already in `package.json`, unused). The retention gap the audit calls "highest-value single build."
- **1.3 Real `deleteAccount` end-to-end test** in a prod-like env. PR 6 removed the fake branch; PR 6 did not verify the real path against real cascade behavior. Worth a scripted test before beta.
- **1.4 AI rate limit + Sentry + leaked-password toggle + SECURITY DEFINER function review.**
- **1.5 Field-level journal encryption decision** — see `docs/compliance-readiness.md` §164.312(a)(2)(iv). Business decision (server-side, client-side, or explicitly none) before code.
- **1.6 Insight guardrails** — min-data thresholds + feedback buttons. Table in the plan.
- **Phase 2** — beta invites + Playwright E2E for core flows (currently only auth/legal covered) + verified-mindmap cron enable + ADR-001.

## Gap-audit findings we DID NOT fix

From my session doc-review pass, these went into the gap-audit but were not selected as PR chunks:

- **G7 `getBodySensations` orphaned** — likely tied to an unfinished body-map ↔ check-in integration. Worth 20 minutes to trace and decide: wire or delete.
- **G8 `getInsightHistory` orphaned** — the "trend over time" insight affordance doesn't exist. Same choice: build the UI or delete the action.
- **G10 Orphan trees** — `legacy/`, `cursor-claude-legacy/`, `sleep-stress/` at repo root are all unreferenced. `sleep-stress/` in particular is a whole second `frontend/`. Add to `.gitignore` or `git rm -r`.

## Branches to leave alone

`feat/ml-graph-*` and `feat/ml-verifier-quality` are yours from earlier
merges into main (checked git log — they're the four leading up to
`6544637`). No action needed on them.

`ui-concept-prod` and `ui-polish` are branches you had before this session
(ahead/behind their tracked remotes); untouched by me.

The throwaway `verify/all-prs` local branch is not pushed and can be
deleted with `git branch -D verify/all-prs` when you feel like it.

## How to resume

Simplest read-in: skim [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md) if it's
still open in your IDE, or the plan doc that lives on `fix/security-hygiene`
as [LAUNCH_READINESS_AUDIT.md](../LAUNCH_READINESS_AUDIT.md), then this file.
The auto-memory file `beta-launch-plan-2026-08.md` also carries the current
state.

Pick up wherever fits your energy — the merges above are the smallest
next step; the Phase-1 list is the biggest. My take: **merge PR 6 (fake
deletion) today**, then run the four low-risk PRs through Vercel preview,
then decide whether to do 1.4 (Sentry + rate limit + leaked-password
toggle) before or after the merge parade.
