// Sentry for the Node.js runtime (server actions, API routes, RSC).
// Health app: never send PII, never attach request bodies — journal text,
// check-in payloads, and transcripts must not reach a third party.
import * as Sentry from "@sentry/nextjs";

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.VERCEL_ENV ?? process.env.NODE_ENV,
  tracesSampleRate: 0.1,
  // No PII: with this off the SDK does not attach request bodies, cookies,
  // headers with auth material, or user IPs.
  sendDefaultPii: false,
});
