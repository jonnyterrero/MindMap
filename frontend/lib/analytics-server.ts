import "server-only"

import { PostHog } from "posthog-node"
import type { AnalyticsEventName, AnalyticsProps } from "@/lib/analytics-events"
import { createClient } from "@/lib/supabase-server"
import { resolveAnalyticsOptIn } from "@/lib/analytics-consent"

async function hasAnalyticsConsent(userId: string): Promise<boolean> {
  const supabase = await createClient()
  const { data: preference, error: preferenceError } = await supabase
    .from("user_privacy_settings")
    .select("analytics_opt_in")
    .eq("user_id", userId)
    .maybeSingle()

  if (preferenceError) return false

  let consent = null
  if (!preference) {
    const result = await supabase
      .from("consent_records")
      .select("consent_given")
      .eq("user_id", userId)
      .eq("consent_type", "analytics_collection")
      .order("created_at", { ascending: false })
      .limit(1)
      .maybeSingle()
    if (!result.error) consent = result.data
  }

  return resolveAnalyticsOptIn(preference, consent)
}

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
  try {
    if (!(await hasAnalyticsConsent(distinctId))) return
    const client = getClient()
    if (!client) return
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
