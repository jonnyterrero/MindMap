# MindMap — Compliance Readiness Inventory

**As of:** 2026-08-06 (immediately after PR 6-10 land).
**Author:** Written from a codebase read-through, not a legal opinion. Get one before you rely on this to make a compliance claim.
**Purpose:** Answer, at any moment: "What would it take to say we're HIPAA / SOC 2 aligned?" without touching either right now.

The Beta Launch Plan freezes both frameworks until Phase 3 (60-90 days after the beta gate closes). This doc is the paper trail for that decision, plus the map for undoing it later.

---

## TL;DR

- **HIPAA does not automatically apply to MindMap today** because there is no Covered Entity or Business Associate contract obligating it to. The moment a clinic, insurer, or employer wellness program integrates and we handle their PHI on their behalf, HIPAA applies to that data flow. Until then, "HIPAA compliant" is a marketing choice, not a legal state.
- **We must never claim HIPAA compliance in marketing or in-app copy** while the gaps below exist. The plan already forbids this; PR 6 (fake-deletion removal) and PR 10 (honest wearables copy) closed two paths that hinted otherwise. Grep the codebase for `HIPAA` before every deploy that touches marketing text.
- **The single most expensive gap is field-level encryption for journal content.** Everything else on this list is weeks; that is months, and only useful if we also decide who holds the keys.
- **SOC 2 is a bigger org lift than HIPAA at our size.** It doesn't come from code alone; it comes from written policies, audits, quarterly access reviews, and a paid auditor. Do not attempt without an ADR.

---

## When each framework actually applies

### HIPAA — 45 CFR §164
Applies when MindMap is either:
1. A **Covered Entity** — a healthcare provider that bills electronically, a health plan, or a clearinghouse. We are not; we're a self-tracking consumer app.
2. A **Business Associate** — we handle Protected Health Information on behalf of a Covered Entity. This kicks in the day a clinic, insurer, or provider organization signs a **Business Associate Agreement (BAA)** with us. Signing a BAA before we can meet the technical requirements is malpractice.
3. Under a state law that piggybacks on HIPAA definitions (California, Texas, others). Rare for consumer wellness apps but worth checking with counsel per state before B2B.

Self-reported mood, sleep, migraine, and journal content **from an individual to themselves** is **not PHI under HIPAA** even though it is obviously health-adjacent. It becomes PHI the moment it flows through a HIPAA-covered relationship — for example, if a clinician's dashboard reads a patient's MindMap data as part of clinical care.

### SOC 2 — AICPA
Applies whenever an enterprise B2B buyer requires it. The five Trust Services Criteria:
- **Security** (mandatory for any SOC 2)
- **Availability**
- **Confidentiality**
- **Processing Integrity**
- **Privacy**

A **Type I** report is a point-in-time attestation. **Type II** covers a period (typically 6-12 months) and is what enterprise buyers ask for. Type II is what "SOC 2 compliant" usually means in practice.

SOC 2 tests controls, not code. About 70% of getting a report is written policy, evidence collection, and org process; about 30% is technical.

---

## Vendor & data inventory

Every third party MindMap touches with user data, and its current data-handling status:

| Vendor | Role | Data it sees | BAA available? | BAA in place? |
|---|---|---|---|---|
| **Supabase** | Auth, Postgres, storage, cron | Everything: emails, self-reported health data, journal plaintext | Yes, on **Team plan** ($599/mo minimum). Not on Free. | No |
| **Vercel** | Hosting, edge, CI | HTTP metadata; Server Components may hold PHI briefly in memory during a request | Yes, on **Enterprise plan**. Not on Pro. | No |
| **Anthropic** | AI chat, journal reflection, report generation, voice sentiment | Journal excerpts, insight text, medication names in prompts | Yes (Anthropic offers BAAs). | No |
| **Open-Meteo** (weather) | Public weather API | Coordinates (`weather_lat`/`weather_lon`) only, on user opt-in | No BAA — public unauthenticated API | N/A (do not send PHI) |
| **Vercel Analytics** | Product analytics | Page path, referrer, viewport, coarse geo | Not applicable | N/A (should be off for PHI-scoped users) |
| **Cloudflare** | Registrar + DNS for `getmindmapplus.app` | DNS metadata only, does not proxy | Not applicable | N/A |

**Not yet integrated** (in `package.json` or planned): `@capacitor/local-notifications`, `@capacitor/push-notifications` (0.6 in the plan) — if we ever route notification content through a third-party push service (APNs/FCM directly is fine; Pusher/OneSignal is not without a BAA), add them to this table first.

### Data classification (as of this audit)

The `profiles` table + these health-adjacent tables carry the sensitive columns:

```
mindmap_entries              — mood, anxiety, depression, sleep, migraine, notes (free text)
mindmap_journal_entries      — title, content (free text), mood_tags, is_private
mindmap_medications           — name, dosage, notes
mindmap_body_sensations      — pain locations + intensity + notes
mindmap_crisis_events         — severity, keyword, tone (see crisis-detection.ts)
mindmap_ai_conversations      — free-text back-and-forth with Claude
mindmap_ai_reports           — AI-generated summary_markdown
mindmap_wearable_data        — HRV, sleep score (manual entry today)
consent_records              — consent given, timestamped
security_audit_events         — provider-access audits
```

Journal `content` is the highest-risk field because it's free text and unbounded — someone will write things there they wouldn't tell their doctor. **This is the one field where the encryption-at-rest posture matters most, and today it stores plaintext.**

---

## HIPAA Security Rule (§164.312) technical safeguards — gap analysis

Each row: what the rule requires, what MindMap has today, and what closes the gap.

### §164.312(a)(1) Access control
> Assign a unique name and/or number for identifying and tracking user identity.

- ✅ Supabase auth issues unique UUIDs (`auth.uid()`); every table with health data has `user_id` FK; every RLS policy is scoped to `auth.uid()`. Verified by the 100-lint Supabase advisor sweep in `beta-launch-plan-2026-08` — 0 ERROR-level, 0 missing-RLS.
- ⚠️ Emergency access procedure — no documented "break-glass" flow for a support engineer to view a specific user's data (nor is one currently used). Formalize before any B2B/BAA relationship.
- ⚠️ Automatic logoff — Supabase session tokens expire on their own schedule; no explicit re-auth on sensitive actions (deletion is protected by "type DELETE" but not by password re-prompt). Password re-prompt exists for password change (`settings/actions.ts::changePassword`). Extending it to deletion is 20 lines.

### §164.312(a)(2)(iv) Encryption/decryption
> Implement a mechanism to encrypt and decrypt electronic protected health information.

- ✅ Supabase Postgres is encrypted at rest (AES-256 disk-level, managed by AWS RDS/EBS).
- 🔴 **Field-level encryption on journal content is not implemented.** Migration 007 added `body_encrypted`, `encryption_key_id`, `encryption_algo`, `is_encrypted` columns to `mindmap_journal_entries` but nothing writes to them; new rows go into plaintext `content`. This is the audit's #5 and the plan's 1.5. Options:
  - **Envelope encryption** — a per-user data key encrypted by a KMS master key (AWS KMS, Google Cloud KMS, HashiCorp Vault). Server can decrypt; user cannot lose access.
  - **Client-side (zero-knowledge) encryption** — a key derived from the user's password never leaves their device. Server can't decrypt. Blocks server-side features that read the plaintext: AI reflection, crisis detection, AI report generation.
  - **Do neither, document what disk-level covers, don't claim more.** The current honest state.

The three are not equivalent. Envelope encryption is what almost every health app labeled "HIPAA-compliant" actually does. Zero-knowledge is stricter and breaks the AI features that are half the product. **Deciding which of the three we want is a business decision, not an engineering one; the code can implement any of the three in about 2 weeks each once decided.**

### §164.312(b) Audit controls
> Implement hardware, software, and/or procedural mechanisms that record and examine activity in information systems that contain or use electronic protected health information.

- ✅ `security_audit_events` table exists (migration 011) with an immutable-row trigger (`fn_security_audit_immutable`).
- ✅ `fn_log_provider_access` writes an audit row every time a provider RPC reads a patient's data. Verified live.
- 🟠 **Ordinary user reads of their own data are not audited.** Under strict HIPAA interpretation, every access to a PHI record — including a user reading their own data — should be logged. For a self-tracking app, the risk-vs-noise tradeoff usually settles on "no, but document the decision"; that documentation has to exist.
- 🟠 **No log aggregation.** Supabase logs are per-project; there's no central retention. HIPAA requires 6 years of audit-log retention. Currently we hit that only accidentally by never deleting.

### §164.312(c)(1) Integrity
> Protect ePHI from improper alteration or destruction.

- ✅ ON DELETE CASCADE FKs (migration 004) ensure child rows go with parents.
- ✅ Immutable audit-row trigger.
- ⚠️ No content hashing / signature on `mindmap_ai_reports` or `mindmap_predictions` outputs. A tampered AI report from a service-role compromise wouldn't be detected. Low priority; think about it after Sentry lands.

### §164.312(d) Person or entity authentication
> Verify the identity of the person or entity seeking access.

- ✅ Supabase email/password auth with password change requiring re-verification.
- 🔴 **Leaked-password protection is OFF** in Supabase (advisor lint, still open — plan item 1.4). Flipping it enables HaveIBeenPwned check on new passwords. Free, one toggle.
- 🟠 **No MFA option.** Supabase supports TOTP MFA via `auth.mfa.*`. Not enabled. Consumer wellness generally doesn't require MFA; anyone offering it for enterprise/B2B usually does.

### §164.312(e)(1) Transmission security
> Guard against unauthorized access to ePHI transmitted over a network.

- ✅ Vercel enforces HTTPS end-to-end on all hosted pages.
- ✅ Anthropic and Supabase calls go over TLS by default.
- ✅ Middleware sets no known-bad response headers; the /sw.js route sets `Cache-Control: no-store` on the service worker (PR 3).
- 🟠 **No HSTS header set explicitly** — Vercel sets a default; verify it says `max-age=31536000; includeSubDomains; preload`. If we ever run our own edge middleware to strip it, we've regressed.

---

## HIPAA Privacy Rule + Breach Notification — organizational gaps

These don't show up in code but are hard requirements the day HIPAA applies.

- 🔴 **No Notice of Privacy Practices** document (HIPAA §164.520). The existing `/privacy` page is a privacy policy, not an NPP. Different requirement.
- 🔴 **No breach notification playbook.** §164.400 requires notice within 60 days of discovery to affected individuals, to HHS, and (for breaches > 500 people) to the media. We have no template, no rehearsal, no chain-of-command doc.
- 🔴 **No signed BAAs on our vendor list** (see table above). Signing one enterprise BAA before we have BAAs with our own vendors is a legal exposure — we'd be promising controls we haven't secured.
- 🟠 **No formal risk analysis** (§164.308(a)(1)). This doc is a start; a real one is a workshop with an outside assessor and is what most engineering teams underestimate.
- 🟠 **No workforce training program.** Applies once we have employees or contractors with data access.

---

## SOC 2 (Type II) mapping — brief

SOC 2 measures controls against 5 Trust Services Criteria. Only Security is mandatory. Each criterion needs 3-15 documented, evidenced controls. The typical minimum for a startup Type II is 60-80 controls total.

Rough coverage today, per criterion:

| Criterion | Controls with technical evidence | Controls with documented policy | Est. effort to close |
|---|---|---|---|
| Security | ~40% (RLS, HTTPS, backups, dep audit gate after PR 7) | ~5% (no written policies) | 3-6 months |
| Availability | ~30% (Vercel SLAs, Supabase replication) | 0% | 2-3 months |
| Confidentiality | ~20% (RLS, encryption at rest) | 0% | 3-4 months |
| Processing Integrity | Rules engines are unit-tested (39 tests, `frontend/tests/`); ML has 221 (`ml/tests/`); no formal change-management policy | 0% | 3 months |
| Privacy | Consent flow, export, deletion via `deleteAccount` | 5% | 3 months |

Beyond code: a Type II report needs **6 months of evidence** collected against every control. Meaning: the calendar starts when the policies are written and the tooling is in place, not before. **Sequential steps for a Type II timeline: policies (1-2mo) → tooling & remediation (2-3mo) → observation period (6mo) → audit (1-2mo). Total ~12 months minimum.**

Cost: expect $15-40k for the auditor plus $12-40k/yr for a compliance-tooling platform (Vanta, Drata, Secureframe) if we don't want to hand-roll it. **This is why the plan freezes SOC 2 until we have real B2B revenue signal.**

---

## Recommended sequencing

Two paths, cheapest first:

### Path A — "Stay a consumer app; be honestly good at it"
1. **Never claim HIPAA / SOC 2 alignment in marketing.** (Ongoing.)
2. Ship the plan's Phase 1 (Sentry, AI rate limit, real deletion, encryption posture decision, insight thresholds) — closes half of §164.312 as a side effect.
3. Enable leaked-password protection (one Supabase toggle).
4. Formalize the "we do not encrypt at field level; disk-level only" decision in `docs/security-posture.md` and link it from the privacy page.
5. Never sign a BAA.

Cost: ~1 sprint of engineering on top of the existing plan. Ceiling: consumer app, no B2B, no enterprise buyer.

### Path B — "First B2B / clinic pilot is on the horizon"
Do A, then:
6. Upgrade to Supabase Team + Vercel Enterprise + Anthropic BAA. Sign BAAs with each. (~$1k/mo ongoing.)
7. Implement envelope encryption for `mindmap_journal_entries.content` (2-3 weeks). Decide server-side vs client-side; document.
8. Add password re-prompt on delete-account and on data-export. Consider TOTP MFA for provider-role accounts.
9. Extend the audit log: retention + rotation policy; every provider RPC and (optionally) `.select` from a health-adjacent table logs.
10. Write the NPP, breach playbook, workforce training doc.
11. Retain a HIPAA-savvy attorney to review.

Cost: ~1-2 quarters, most of it in step 10. Ceiling: BAA-signable, but not yet SOC 2.

### Path C — "Enterprise deal requires SOC 2 Type II"
Do A + B, then:
12. Adopt a compliance platform (Vanta / Drata / Secureframe). This is a business-tooling decision more than an engineering one.
13. Write policies: incident response, change management, access review, vendor management, backup/restore, disaster recovery, business continuity. Store them somewhere versioned.
14. Start the observation period. Live for it. Every quarterly access review, every incident postmortem, every off-boarded contractor must have a paper trail.
15. Book an auditor 3-6 months ahead.

Total time: ~12 months from Path A. Total cost: probably ~$60-100k first year including auditor + platform + engineering time. **Do not start without a signed enterprise deal that requires it and covers the cost.**

---

## What to do next

- Nothing in this doc is a work item until the freeze lifts.
- When the freeze lifts, this doc's TL;DR + the vendor table + the §164.312 gaps are the input to an ADR (`docs/decisions/adr-002-lift-compliance-freeze.md`).
- Until then: re-read the "Never claim" line before any marketing copy change.

Related: [beta-launch-plan (memory)](../memory-not-in-repo), [MindMap+ Launch-Readiness Audit](../LAUNCH_READINESS_AUDIT.md) (on `fix/security-hygiene`), [Supabase security advisor sweep](../supabase/migrations/023_rls_policy_fixes.sql) (results in commit body of PR 2).
