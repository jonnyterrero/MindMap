# MindMap+ — Launch-Readiness Audit

**As of:** July 2026 (`main` @ `fd38f68`)
**Scope:** What's missing / incomplete before launching as a fully-fledged app, from a *launch-with-real-users* lens.
**Method:** Grounded in the actual codebase — route/API enumeration, feature-wiring checks, Supabase security advisors, and reads of the load-bearing files. Point-in-time; re-verify before acting.

**Severity legend:** 🔴 Critical · 🟠 High · 🟡 Medium · 🟢 Solid (no action)

---

## 🔴 Critical — likely explains the loudest feedback

### 1. No reminders or notifications at all
`@capacitor/local-notifications` and `@capacitor/push-notifications` are in `package.json` but **never imported or wired anywhere** — nothing schedules a check-in reminder, medication nudge, or re-engagement prompt. For a daily-tracking mental-health app this is the missing habit loop.
- **Maps to feedback like:** "I forget to use it," "it doesn't pull me back in."
- **Verify:** no `LocalNotifications.schedule` / `PushNotifications.register` calls exist in `app/`, `lib/`, `components/`, `hooks/`.
- **Fix:** wire local notifications (check-in + med schedule reminders) at minimum; push for server-driven nudges.

### 2. The `/api/v1/*` API is a fake facade
`/api/v1/mood`, `/sleep`, `/medications`, `/analytics` return **hardcoded sample data** (e.g. `"Feeling great today"`, `2024-01-15`) and their "auth" only checks that *some* `x-api-key` header exists — any value passes, no user scoping, no database. Code comment: *"In a real app, fetch from database / For now, return sample data structure."*
- **Risk:** misleading if advertised as an integration API; the "auth" is trivially bypassed.
- **Fix:** build for real (validate keys, scope to user, hit DB) **or delete the routes**.

### 3. Wearables can't actually connect to anything
Data model exists (`mindmap_wearable_sources`, `mindmap_wearable_data`) and the UI reads it, but there's **no device integration** — no Apple Health / Fitbit / Oura / Google Fit OAuth — and the only ingestion path is the stub API in #2.
- **Net:** "wearables" is a settings screen + an empty table with no real way to get data in.
- **Fix:** integrate a real source (Apple Health via native bridge, or an aggregator like Terra/Rook), or reframe the feature.

---

## 🟠 High — launch-blocking for a health app specifically

### 4. Account deletion doesn't delete
`deleteAccount` only inserts a row into `data_deletion_requests`; nothing processes it. Right-to-erasure (GDPR/CCPA) needs real deletion or a documented, honored manual process.

### 5. Journal content is stored as plaintext
Encryption columns (`body_encrypted`, `encryption_key_id`, `encryption_algo`) exist but are **unwired** — journals store plaintext in `content`. Supabase encrypts at rest at the disk level; whether that's sufficient is an explicit risk/claims decision, not a default.

### 6. No error monitoring
No Sentry or equivalent anywhere — production exceptions are invisible and un-triageable. Highest-value ops add before real users.

### 7. AI chat has no rate limiting
`/api/ai-chat` is authenticated and token-capped (`max_tokens: 1024`, history capped at 20) but nothing limits request volume → cost and abuse exposure on a paid Anthropic endpoint.

### 8. Key rotation outstanding
Prod Supabase/Anthropic keys were pasted into a chat session earlier and should be rotated. Also gates enabling the ML graph cron.

---

## 🟡 Medium — quality, trust, polish

### 9. ML is rules-only on synthetic data
Insights/predictions are conservative **rule engines** (`prediction-engine.ts`, `correlation-engine.ts`, `insights-engine.ts`) — no trained models, no real validation set. Fine *if* framed as "early/experimental reflection" (the copy mostly does), but it isn't the ML the branding implies.
- The verified-**Mindmap view is empty in prod** because the graph cron secrets (`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, optional `ANTHROPIC_API_KEY`) aren't set.
- The ~200-entry real, dual-annotated gold set (the trust gate) is still unbuilt — needs human annotation.

### 10. No PDF report export
The reports cron produces data, not a document. A clinician-shareable PDF is a common ask and isn't there (no `puppeteer` / `react-pdf` / `jsPDF`).

### 11. Thin end-to-end coverage
Playwright covers **auth + legal pages only** (`auth-routes`, `authenticated`, `legal-pages`). Core flows — check-in → journal → insights → medications — have no E2E. Unit tests exist for the rule engines and crisis detection (good).

### 12. Only one loading skeleton
Just `app/(app)/loading.tsx` — most pages flash blank on slow networks. Add route-level `loading.tsx` / Suspense for the data-heavy pages.

### 13. Offline sync is journal-only
The offline queue (`lib/offline-queue`, `hooks/use-offline-sync`) is real but scoped to **journal writes**. Check-ins and medication logs don't queue offline, so the core daily action fails without connectivity on mobile.

### 14. Two quick DB hardening items
- **Leaked-password protection is OFF** (one Supabase toggle — enable HaveIBeenPwned check).
- **4 SECURITY DEFINER functions executable by authenticated users** — review for privilege-escalation.
- (Minor) one extension installed in the `public` schema.

### 15. English only
No i18n scaffolding (`next-intl` / `react-intl` absent). Fine for launch scope, but note it for reach.

---

## 🟢 Genuinely solid (no action — for balance)

- **RLS posture is clean:** 0 ERROR-level Supabase security advisors across 94 checks; no missing-RLS or exposed tables.
- **Error boundaries** exist app-wide (`global-error.tsx` + per-route).
- **Data export** works (`exportUserData`).
- **Crisis detection** logic + `/crisis-resources` page are real (unit-tested).
- **Medical disclaimers**, `/ai-disclosure`, ToS, privacy pages in place and styled.
- **Verified-mindmap safety architecture** (generation ≠ verification, fail-closed, provenance) is thoughtful.
- **Installable PWA** (`manifest.json`, `sw.js`); offline journal queue functions.

---

## Suggested sequencing against feedback

| If the feedback is about… | Start with |
|---|---|
| Retention / forgetting to log | **#1 Notifications & reminders** |
| "It doesn't do what it says" | **#2 stub API**, **#3 wearables**, **#9 ML framing** |
| Launch safety / trust | **#4–#8** (deletion, encryption, monitoring, rate limiting, keys) |
| Polish complaints | **#10–#13** (PDF, E2E, loading states, offline scope) |

**Highest-leverage single build:** #1 (notifications) — biggest behavioral gap, and a concrete build rather than a judgment call.

**Can't be closed in code (needs a human):** legal review of disclaimers/ToS, the encryption/HIPAA posture decision, and the real-data ML validation (annotation).
