# MindMap — Mobile (Capacitor) Guide

MindMap ships to iOS and Android via **Capacitor in hosted/remote mode**: the
native app is a thin native shell whose WebView loads the live production site
(`https://getmindmapplus.app`). The Capacitor bridge still gives us native
plugins (push notifications, health data, haptics, status bar, etc.).

## Why hosted mode (not static export)

MindMap uses Next.js Server Actions, middleware-based auth/consent gating, and
Supabase SSR cookies — all of which require a server. `next build` cannot be
statically exported (`output: 'export'`), so we point Capacitor's `server.url`
at the deployed app instead of bundling static HTML.

**Big upside:** web deploys update the mobile app instantly. You only rebuild
and resubmit the native app when the **native shell** changes (new plugin,
config, icons, OS target) — not for normal web/UX changes.

`mobile/www/index.html` is only an offline/cold-start fallback; the WebView
normally loads `server.url`.

---

## One-time setup (on a Mac)

Prerequisites:
- **iOS:** macOS, Xcode + Command Line Tools, CocoaPods (`sudo gem install cocoapods`), an Apple Developer account.
- **Android:** Android Studio + JDK 17, an Android SDK, a Google Play Developer account.

```bash
cd frontend
npm install                 # installs @capacitor/* from package.json
npm run cap:add:ios         # creates ./ios   (needs Xcode/CocoaPods)
npm run cap:add:android     # creates ./android (needs Android Studio)
npm run cap:sync            # copies config + plugins into the native projects
```

`config` is already written (`capacitor.config.ts`): appId `com.heartwire.mindmap`,
appName `MindMap`, `server.url = https://getmindmapplus.app`.

> The generated `ios/` and `android/` folders are large native projects. Decide
> whether to commit them or gitignore them (most teams gitignore and regenerate
> via `cap add` + `cap sync` in CI/EAS-style flows).

## Dev loop

```bash
npm run cap:sync            # after any capacitor.config.ts / plugin change
npm run cap:open:ios        # opens Xcode  -> run on simulator/device
npm run cap:open:android    # opens Android Studio -> run on emulator/device
```

Because of hosted mode, you usually just need a network connection — the shell
loads the live site. No `npm run build` is required to see web changes on device.

---

## Native integrations (wired to existing backend)

### Push notifications → `device_push_tokens` / `notification_delivery_log`
The backend already has `device_push_tokens` (platform, provider, token) and
`notification_delivery_log`. On the client:
1. Register with `@capacitor/push-notifications`, request permission, get the
   device token in the `registration` listener.
2. POST it to a new route/action that upserts into `device_push_tokens`
   (`provider`: `apns` for iOS, `fcm` for Android).
3. iOS needs an APNs key in Apple Developer + Push Notifications capability in
   Xcode; Android needs a Firebase project + `google-services.json`.
A server worker then sends via APNs/FCM and logs to `notification_delivery_log`.

### Health / wearables → `mindmap_entries`
Use a Capacitor health plugin (e.g. `@capacitor-community/health-connect` for
Android Health Connect, and a HealthKit plugin for iOS) to read **sleep, HRV,
resting HR, steps**. Map them into the existing `mindmap_entries` columns
(`sleep_minutes`, `hrv`, …) via a server action, with explicit user opt-in
(reuse the Settings opt-in pattern from weather/AI). This replaces the web-only
CSV import path. Requires HealthKit entitlement (iOS) and Health Connect
permissions (Android) — declare data types and add privacy strings.

### Other useful plugins (already in package.json)
- `@capacitor/app` — deep links / app URL open (Supabase email confirm, password reset).
- `@capacitor/preferences` — small native key/value (e.g. onboarding hints).
- `@capacitor/status-bar`, `@capacitor/splash-screen`, `@capacitor/haptics`, `@capacitor/network`.

---

## Store submission (high level)

Bundle identity: **`com.heartwire.mindmap`** / **MindMap**.

- **iOS (TestFlight → App Store):** set signing in Xcode, fill the App Privacy
  "nutrition label" (health data, no tracking), add usage-description strings
  for notifications + HealthKit, screenshots, privacy policy URL
  (`/privacy`), then upload + submit for review.
- **Android (Internal testing → Play):** build a signed `.aab`, complete the
  Data Safety form + Health Apps declaration, add the privacy policy URL,
  screenshots + feature graphic, then roll out.

Keep store copy **wellness/self-tracking only** — no diagnosis/treatment claims
(the in-app `MedicalDisclaimer` already enforces this tone).

---

## Deep-link note (auth redirects)

Supabase email-confirmation and password-reset links point at
`https://getmindmapplus.app/...`. In hosted mode these open inside the WebView
fine. If you later add custom URL schemes / universal links, register them in
Xcode (Associated Domains) and `AndroidManifest.xml`, and add the scheme to
`allowNavigation` in `capacitor.config.ts`.
