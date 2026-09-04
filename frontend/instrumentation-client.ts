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
    // Health app: only build person profiles for signed-in users.
    // Autocapture never records input values. Session replay stays off so
    // journal text and check-in scores cannot leak through the recorder.
    person_profiles: "identified_only",
    disable_session_recording: true,
  })
}

// Tie events to the Supabase user via their pseudonymous UUID (no email/PII).
if (
  POSTHOG_KEY &&
  process.env.NEXT_PUBLIC_SUPABASE_URL &&
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
) {
  void import("@/lib/supabase").then(({ createClient }) => {
    createClient().auth.onAuthStateChange((event, session) => {
      if (event === "SIGNED_OUT") {
        posthog.reset()
      } else if (session?.user) {
        posthog.identify(session.user.id)
      }
    })
  })
}
