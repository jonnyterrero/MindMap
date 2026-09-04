# MindMap — Apple App Store Submission Checklist

**As of:** 2026-09-04. Companion to `frontend/MOBILE.md` (build mechanics).
Everything below is grounded in the app's actual behavior — answers to Apple's
questionnaires should not drift from this without a code change to justify it.

Bundle: `com.heartwire.mindmap` · Name: **MindMap** · Hosted-mode Capacitor
shell loading `https://getmindmapplus.app`.

---

## 1. What's already prepared in the repo (done)

| Item | Where |
|---|---|
| Capacitor config (hosted mode, allowNavigation, splash, push presentation, local-notification icon) | `frontend/capacitor.config.ts` |
| App icon 1024×1024 + splash 2732×2732 (light/dark) for `npx @capacitor/assets` | `frontend/resources/` — **note:** icon is upscaled from the 512px PWA icon; replace with a true 1024 source before submission if available |
| Apple privacy manifest (no tracking; health/sensitive/journal/email collected, linked, app-functionality only) | `frontend/mobile/ios/PrivacyInfo.xcprivacy` — copy to `ios/App/App/` after `cap add ios` |
| Local notifications (check-in / med / journal reminders) | wired end-to-end (settings UI → `mindmap_reminders` → device schedule) |
| In-app account deletion (App Store Guideline 5.1.1(v) requires it) | Settings → Data & privacy → real `deleteAccount` |
| Privacy policy, ToS, medical disclaimer, AI disclosure pages | `/privacy`, `/terms`, `/medical-disclaimer`, `/ai-disclosure` (linked from signup + consent) |
| Crisis resources page + in-chat crisis detection | `/crisis-resources`, `lib/crisis-detection.ts` |

## 2. Mac-side build steps (cannot be done on Windows)

```bash
cd frontend
pnpm install
pnpm cap:add:ios          # generates ios/ (Xcode + CocoaPods required)
npx @capacitor/assets generate --ios   # icons/splash from resources/
cp mobile/ios/PrivacyInfo.xcprivacy ios/App/App/   # add to App target in Xcode
pnpm cap:sync
pnpm cap:open:ios
```

In Xcode:
- [ ] Set the signing team; bundle id `com.heartwire.mindmap`.
- [ ] Capabilities: **Push Notifications** only if/when push ships — local
      notifications need **no** capability or usage string.
- [ ] Deployment target: Capacitor 8 default (iOS 14+) is fine.
- [ ] Archive → upload to TestFlight.

## 3. App Store Connect — App Privacy ("nutrition label")

Declare exactly this (matches `PrivacyInfo.xcprivacy` and the DB):

| Data type | Collected? | Linked to identity | Tracking | Purpose |
|---|---|---|---|---|
| Health & Fitness (mood, sleep, symptoms, meds) | Yes | Yes (account) | No | App functionality |
| Sensitive info (mental-health check-ins) | Yes | Yes | No | App functionality |
| User content (journals, voice transcripts, AI chat) | Yes | Yes | No | App functionality |
| Email address | Yes | Yes | No | App functionality (auth) |
| Product interaction (PostHog/Vercel analytics, pseudonymous UUID, no PHI in payloads) | Yes | Yes | No | Analytics |
| Precise location | **No** (weather uses user-entered coarse location, opt-in) | — | — | — |
| Advertising / tracking data | **No** | — | — | — |

- [ ] "Do you or your third-party partners use data for tracking?" → **No.**
- [ ] Privacy policy URL: `https://getmindmapplus.app/privacy`.

## 4. Review-sensitive points for a mental-health app

- [ ] **Guideline 1.4.1 (physical harm):** store copy and screenshots must stay
      wellness/self-tracking — no diagnosis, treatment, or "detects migraines"
      claims. The in-app `MedicalDisclaimer` tone is the ceiling.
- [ ] **Guideline 5.1.1(v):** account deletion in-app — already real; verify the
      full cascade once in TestFlight before submission (Beta plan 1.3).
- [ ] **AI content:** `/ai-disclosure` exists; mention it in Review Notes and
      note the crisis-detection + 988 guidance, AI daily quotas, and that AI
      output is never medical advice.
- [ ] **Login for review:** create a demo account with seeded data and put the
      credentials in App Review notes (reviewers must reach `/today`,
      `/insights`, `/companion` without signing up).
- [ ] **Export compliance:** only standard HTTPS/TLS → answer "uses exempt
      encryption" (`ITSAppUsesNonExemptEncryption = NO` in Info.plist).
- [ ] Age rating questionnaire: expect 12+ (infrequent/mild medical info).

## 5. Assets still needed (human)

- [ ] 6.7" and 6.5" iPhone screenshots (5 each) — capture from simulator on
      the seeded demo account; suggested flow: Today check-in → Insights →
      Journal → Companion → Settings/reminders.
- [ ] App Store description + keywords + promo text (wellness tone).
- [ ] Support URL (`/support` exists and works).
- [ ] Optional: true 1024px master icon to replace the upscaled one.

## 6. Post-approval

- [ ] Phased release recommended.
- [ ] Remember hosted mode: web deploys update the app instantly; only native
      shell changes (plugins, config, icons) need a new build + review.
