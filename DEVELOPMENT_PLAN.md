# MindMap — Development Plan (Beta)

**Sources:** Deep Audit artifact (2026-07-14) + Combined Next-Steps Plan (Launch Audit + Curtis rundown, 2026-08-06).
**Verified against:** `main` @ `6544637` on 2026-08-06.
**North star:** Ship a trustworthy private beta — users understand the product, safely sign up, complete a daily check-in, build a 10-day baseline, and see non-medical pattern insights. Clinical/provider is a **future, separate product** — do not claim it exists.

> How to use this file with Claude Code: work top-to-bottom. Each numbered step is sized to be one prompt/PR. Check items off as they land. File paths are relative to repo root unless noted; frontend paths are inside `frontend/`.

---

## Already done — do NOT redo

Verified in current code:

- [x] Vestigial `frontend/package-lock.json` removed (pnpm-only)
- [x] Landing page at `/` exists, public and session-aware (`frontend/app/page.tsx` with hero, trust items, footer, Supabase session check)
- [x] Graph persistence migration exists (`supabase/migrations/022_mindmap_graphs.sql`)
- [x] Top-level route loading state (`frontend/app/(app)/loading.tsx`)

Also preserve (working — don't rebuild): RLS on all tables, error boundaries, `exportUserData`, crisis detection + `/crisis-resources`, legal pages, verified-mindmap fail-closed architecture, PWA + offline journal queue.

---

## Phase 0 — Security & trust gate (this week)

### 0.1 Rotate exposed production keys — MANUAL, DO FIRST
Supabase (anon + service role) and Anthropic keys were pasted in a prior chat session.
- [ ] Rotate in Supabase dashboard and Anthropic console
- [ ] Update Vercel env vars (`ANTHROPIC_API_KEY`, Supabase keys, `CRON_SECRET`)
- [ ] Update GitHub Actions repo secrets (`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`) — unblocks ML cron + E2E CI

### 0.2 PR ① `fix/security-hygiene` — delete stub API surface
Stub routes return hardcoded fake data; `/api/auth` always returns `{ valid: true }`.
- [ ] Delete `frontend/app/api/v1/analytics/route.ts`
- [ ] Delete `frontend/app/api/v1/medications/route.ts`
- [ ] Delete `frontend/app/api/v1/mood/route.ts`
- [ ] Delete `frontend/app/api/v1/sleep/route.ts`
- [ ] Delete `frontend/app/api/auth/route.ts`
- [ ] Delete `frontend/lib/api-auth.ts` (dead code)
- **Acceptance:** no `/api/v1/*` routes reachable; `pnpm typecheck && pnpm build` pass

### 0.3 PR ② `fix/rls-policy-bugs` — one migration, three verified vulnerabilities
New migration `supabase/migrations/023_rls_policy_fixes.sql` (022 is taken):
- [ ] **Audit-log impersonation:** `security_audit_events` INSERT policy uses `WITH CHECK (auth.uid() = actor_user_id OR auth.uid() IS NOT NULL)` — the second disjunct makes the check a no-op. Drop/recreate without the `OR` clause. (Introduced in migration 011.)
- [ ] **Provider RPC data leak:** `rpc_provider_get_entries()` and `rpc_provider_get_insights()` return `SELECT *` (including free-text `notes`) and check `resource_type` but not `detail_level`. Route through the detail-gated `v_shared_entries_*` views from migration 004, or filter columns explicitly.
- [ ] **Geolocation exposure:** `profiles_provider_read` policy (migration 017) lets any provider with any grant read the full profiles row including `weather_lat`/`weather_lon`. Replace with a restricted view/column list excluding geo columns.
- **Acceptance:** all 3 closed; authorized provider reads still work

### 0.4 PR ③ `chore/env-naming-cleanup`
- [ ] Add `ANTHROPIC_API_KEY` and `CRON_SECRET` to `frontend/.env.example`
- [ ] Rename `frontend/package.json` name `"my-v0-project"` → `"mindmap"`
- [ ] Service worker cache `"mindtrack-v1"` → `"mindmap-v2"` and deduplicate the double SW (`frontend/public/sw.js` vs `frontend/app/sw.js/route.ts` — keep the route)

### 0.5 PR ④ `ci/ml-gate`
- [ ] Add job to `.github/workflows/ci.yml`: `uv run pytest && uv run ruff check . && uv run mypy` in `ml/` on push/PR (177 tests currently pass but only run manually)

### 0.6 Wearables honesty
- [ ] Hide wearable UI or label "Coming soon" — no fake empty integration, no device OAuth in beta (schema stays)

### 0.7 Onboarding funnel completion
- [ ] Signup success → Welcome → Next Steps pages
- [ ] Wire medical disclaimer + AI/ML disclosure + privacy/deletion/support links through the funnel
- [ ] Copy pass: ML framed as "experimental pattern support," never "clinically validated ML"; never claim HIPAA compliance
- Footer short form: `MindMap is a self-tracking tool, not medical advice.`

Required full disclaimer text:
```text
MindMap is for self-tracking, journaling, wellness reflection, and personal pattern discovery only. It is not a medical device and does not diagnose, treat, cure, or prevent any disease or condition. MindMap does not replace professional medical advice, emergency care, therapy, diagnosis, or treatment.

Insights, trends, scores, and predictions are based only on the information you choose to log and may be incomplete or inaccurate. Do not use MindMap to make medication, treatment, or emergency decisions. If you are experiencing a medical emergency or mental-health crisis, contact emergency services or a local crisis hotline immediately.
```

### 0.8 FGCU one-pager (non-code)
- [ ] Positioning + the 12 interview questions (see Validation section)

---

## Phase 1 — Core habit loop + launch-blocking safety (1–2 weeks)

### 1.1 Check-in + baseline UX
- [ ] Polish daily check-in until reliably **< 90 seconds**
- [ ] 10-day baseline plan UX: Day-N progress, expectations, Day-10 report unlock
- [ ] Verify baseline report content generation end-to-end (route exists; generation unverified)

### 1.2 Local notifications (highest technical retention fix)
- [ ] Wire `@capacitor/local-notifications` (already in `package.json`, unused): daily check-in reminder + medication schedule reminders

### 1.3 Real account deletion
- [ ] Test `deleteAccount` end-to-end in prod-like env. Audit says the cascade path exists (`data_deletion_requests` → storage cleanup → `admin.deleteUser` → CASCADE); combined plan says only the request row is inserted. Determine which is true and fix. GDPR/CCPA gate.

### 1.4 AI safety + ops
- [ ] Per-user rate limiting on `frontend/app/api/ai-chat/route.ts` (currently none)
- [ ] Add Sentry (or equivalent) — production is currently blind
- [ ] Supabase: enable leaked-password protection; review the 4 `SECURITY DEFINER` functions

### 1.5 Encryption posture decision
- [ ] `is_encrypted` column exists with no implementation. Either document "disk-level at-rest only" honestly, or wire field-level journal encryption. Do not ship the implied-but-fake state.

### 1.6 Insight guardrails
- [ ] Enforce minimum-data thresholds with "Not enough data yet — complete X more check-ins" states:

| Insight type | Minimum data |
|---|---:|
| Streak / MindMap Score | 1–3 days |
| Sleep trend | 7 days |
| Mood / anxiety trend | 10–14 days |
| Medication consistency | 7–14 days |
| Migraine association | 14–30 days |
| Weather / migraine | 30+ days |
| Personalized predictive risk | 30+ days |

- [ ] Feedback buttons on every insight: helpful / not helpful / inaccurate / unclear
- Architecture rule: `data → sufficiency check → rule/statistical model → confidence threshold → safe template → disclaimer → feedback`. **Never** `data → LLM → health prediction`.

### 1.7 Privacy toggles
- [ ] Analytics opt-out (Vercel Analytics currently loads unconditionally)
- [ ] Confirm `ai-settings.tsx` actually gates all AI calls

### 1.8 Validation (non-code, parallel)
- [ ] 5–10 user interviews + 1–2 faculty; competitor notes (Daylio, Bearable, Migraine Buddy, How We Feel)

---

## Phase 2 — Private beta quality (~30 days)

- [ ] 2.1 Invite 10–20 beta users; instrument check-in completion + onboarding drop-off
- [ ] 2.2 First real in-app 10-day baseline reports (personal — not clinician PDF)
- [ ] 2.3 Extend offline queue beyond journal → check-ins + medication logs
- [ ] 2.4 PR ⑤ `feat/loading-states`: per-route `loading.tsx` skeletons (only `(app)/loading.tsx` exists), custom `not-found.tsx`, empty states for dashboard / insights / medications / routines
- [ ] 2.5 Playwright E2E: check-in → journal → insights → medications; enable the E2E CI job (secrets now set)
- [ ] 2.6 Enable verified-mindmap graph cron (migration 022 already in) — only after key rotation; empty prod view is worse than off
- [ ] 2.7 Security & Compliance Readiness doc: data inventory, vendor list (Supabase, Vercel, Anthropic, weather, email, push, analytics), AI boundary, BAA needs
- [ ] 2.8 ADR-001: harden current Supabase/Vercel stack vs migrate — decide at end of month; no panic AWS migration

---

## Phase 3 — 60–90 days (only if beta gates pass)

- [ ] Real-data ML: ~30–50 users × 6–8 weeks → walk-forward evaluation → only then consider `rules_only` → `ml_assistive`
- [ ] Expand verifier gold set to ~200 dual-annotated entries; replace placeholder evidence citations with peer-reviewed sources
- [ ] Server-driven push notifications (after local reminders prove value)
- [ ] International crisis resources + locale detection (currently US-only: 988, 741741, 911)
- [ ] PDF export only if beta users demand it; wearables only after retention is proven
- [ ] Provider/B2B track (audit-log writes, versioned consent, FHIR export, RBAC, BAAs, SOC 2) — separate future product

---

## Frozen — do not build now

- New AI chat behaviors / prediction types
- Provider dashboards, hospital workflows, clinician PDF
- Wearable OAuth (Apple Health, Fitbit, Oura, Terra, …)
- Public launch, monetization / paywalls
- SOC 2 chase or AWS/GCP migration without an ADR
- Schema expansion unless security-critical
- i18n (English-only for beta)
- Any "clinically validated ML" claim

---

## Beta gate checklist (complete before inviting beyond a trusted circle)

- [ ] Keys rotated; no secrets in chat/logs
- [ ] Stub `/api/v1` gone
- [ ] RLS policy fixes migrated (audit log, provider RPCs, geo)
- [ ] Account deletion works end-to-end
- [ ] Encryption claims match reality
- [ ] Sentry live
- [ ] AI rate limits live
- [ ] Leaked-password protection on; SECURITY DEFINER functions reviewed
- [ ] RLS re-checked after any schema change
- [ ] No HIPAA-compliant marketing claims
- [ ] Crisis resources reachable; AI does not replace emergency care
- [ ] Vendor list written
- [ ] ADR-001 signed off

---

## First 5 PRs (in order)

| # | Branch | Title | Why first |
|---|---|---|---|
| 1 | `fix/security-hygiene` | Remove stub API routes and dead auth code | Eliminates exploitable endpoints |
| 2 | `fix/rls-policy-bugs` | Patch RLS policy bugs (audit log, provider RPCs, geo leak) | Closes 3 data-leak/impersonation vectors |
| 3 | `chore/env-naming-cleanup` | Fix .env.example, package name, SW cache name | DX + correctness |
| 4 | `ci/ml-gate` | Add ML pytest + ruff + mypy gate to CI | Prevents ML regressions |
| 5 | `feat/loading-states` | Loading skeletons + custom 404 | User-facing polish, low risk |

Key rotation (0.1) happens outside the repo, before PR 1.

---

## Interview questions (validation)

1. What do you use now for mood / symptoms / sleep / meds?
2. What's annoying about those tools?
3. What would make you come back daily?
4. What patterns do you wish you understood?
5. What would make you trust or not trust this?
6. Is a 90-second daily check-in realistic?
7. What would you expect after 10 days?
8. AI summaries: useful or uncomfortable?
9. What data would you refuse to enter?
10. What would you pay for, if anything?
11. What sounds helpful but unnecessary?
12. What's confusing?

---

## Success gates before broader launch

| Gate | Pass condition |
|---|---|
| Understanding | New user can explain what MindMap is/isn't after landing |
| Activation | Completes Day 1 check-in without support |
| Retention | Meaningful % reach 10-day baseline |
| Trust | Deletion, privacy links, disclaimers; no false API/wearable/ML claims |
| Safety | Monitoring, rate limits, keys, crisis path, no medical overclaim |
| Validation | Interviews show real daily-use motivation |
| Ops | Exceptions visible in Sentry; no silent stub endpoints |

> Build the beta cleanly. Design so clinical is possible later. Do not claim clinical exists now.
