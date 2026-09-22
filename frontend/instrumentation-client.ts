// Runs once in the browser before the app hydrates (Next.js 15.3+ convention).
// Pageviews/pageleaves come from `defaults`. This file ships with the GitHub
// repo so every clone and the Vercel production app share the same wiring.
import posthog from "posthog-js"
import * as Sentry from "@sentry/nextjs"

// Sentry error monitoring — inert without a DSN. Health app: no session
// replay (journal text must never reach a recorder), no PII.
const SENTRY_DSN = process.env.NEXT_PUBLIC_SENTRY_DSN
if (SENTRY_DSN) {
  Sentry.init({
    dsn: SENTRY_DSN,
    environment: process.env.NEXT_PUBLIC_VERCEL_ENV ?? process.env.NODE_ENV,
    tracesSampleRate: 0.1,
    sendDefaultPii: false,
  })
}

// Required for navigation instrumentation (no-op when Sentry isn't initialized).
export const onRouterTransitionStart = Sentry.captureRouterTransitionStart

const POSTHOG_KEY = process.env.NEXT_PUBLIC_POSTHOG_KEY
const POSTHOG_HOST =
  process.env.NEXT_PUBLIC_POSTHOG_HOST ?? "https://us.i.posthog.com"

if (POSTHOG_KEY) {
  posthog.init(POSTHOG_KEY, {
    api_host: POSTHOG_HOST,
    defaults: "2026-05-30",
    // Product analytics is explicit opt-in. PrivacyAwareTelemetry enables
    // capture only after resolving the authenticated user's saved preference.
    opt_out_capturing_by_default: true,
    // Re-check Supabase on every page load instead of trusting a stale browser
    // opt-in that may have been revoked from another device.
    persistence: "memory",
    // Health app: only build person profiles for signed-in users.
    // Autocapture never records input values. Session replay stays off so
    // journal text and check-in scores cannot leak through the recorder.
    person_profiles: "identified_only",
    disable_session_recording: true,
  })
}
