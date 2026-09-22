export const ANALYTICS_CONSENT_CHANGED_EVENT =
  "mindmap:analytics-consent-changed";

export type AnalyticsPreferenceRow = {
  analytics_opt_in: boolean;
} | null;

export type AnalyticsConsentRow = {
  consent_given: boolean;
} | null;

/**
 * Prefer the mutable privacy setting. Fall back to the onboarding consent
 * record for accounts created before the settings row exists. Missing data is
 * always treated as an opt-out.
 */
export function resolveAnalyticsOptIn(
  preference: AnalyticsPreferenceRow,
  consent: AnalyticsConsentRow,
): boolean {
  return preference?.analytics_opt_in ?? consent?.consent_given ?? false;
}
