"use client"

import posthog from "posthog-js"
import type { AnalyticsEventName, AnalyticsProps } from "@/lib/analytics-events"

export { AnalyticsEvent } from "@/lib/analytics-events"

export function captureEvent(
  event: AnalyticsEventName,
  properties?: AnalyticsProps,
): void {
  try {
    posthog.capture(event, properties)
  } catch {
    // swallow
  }
}
