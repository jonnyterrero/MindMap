"use client"

import posthog from "posthog-js"
import type { AnalyticsEventName, AnalyticsProps } from "@/lib/analytics-events"

export { AnalyticsEvent } from "@/lib/analytics-events"

export function captureEvent(
  event: AnalyticsEventName,
  properties?: AnalyticsProps,
): void {
  try {
    if (!posthog.has_opted_in_capturing()) return
    posthog.capture(event, properties)
  } catch {
    // swallow
  }
}
