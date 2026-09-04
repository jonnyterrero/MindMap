import { PostHog } from "posthog-node"
import type { AnalyticsEventName, AnalyticsProps } from "@/lib/analytics-events"

function getClient(): PostHog | null {
  const key = process.env.NEXT_PUBLIC_POSTHOG_KEY
  if (!key) return null
  return new PostHog(key, {
    host: process.env.NEXT_PUBLIC_POSTHOG_HOST ?? "https://us.i.posthog.com",
    flushAt: 1,
    flushInterval: 0,
  })
}

/** Fire-and-forget. Never throws — analytics must not break product flows. */
export async function captureServerEvent(
  distinctId: string,
  event: AnalyticsEventName,
  properties?: AnalyticsProps,
): Promise<void> {
  const client = getClient()
  if (!client) return
  try {
    client.capture({
      distinctId,
      event,
      properties: { ...properties, source: "server" },
    })
    await client.shutdown()
  } catch {
    // swallow
  }
}
