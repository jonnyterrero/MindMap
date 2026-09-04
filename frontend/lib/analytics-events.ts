/**
 * Product-analytics event names. Keep payloads free of PHI:
 * no journal text, mood/symptom scores, medication names, emails, or focus areas.
 */
export const AnalyticsEvent = {
  SignedIn: "user_signed_in",
  SignedUp: "user_signed_up",
  SignedOut: "user_signed_out",
  ConsentGranted: "consent_granted",
  OnboardingCompleted: "onboarding_completed",
  CheckinSaved: "checkin_saved",
  JournalCreated: "journal_created",
  CompanionStarted: "companion_started",
  InsightsGenerated: "insights_generated",
  InsightFeedback: "insight_feedback",
  ReportGenerated: "report_generated",
  PwaInstallAccepted: "pwa_install_accepted",
  PwaInstallDismissed: "pwa_install_dismissed",
} as const

export type AnalyticsEventName =
  (typeof AnalyticsEvent)[keyof typeof AnalyticsEvent]

export type AnalyticsProps = Record<string, string | number | boolean>
