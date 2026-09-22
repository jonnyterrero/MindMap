"use client";

import { useCallback, useEffect, useState } from "react";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import posthog from "posthog-js";
import { createClient } from "@/lib/supabase";
import {
  ANALYTICS_CONSENT_CHANGED_EVENT,
  resolveAnalyticsOptIn,
} from "@/lib/analytics-consent";

const POSTHOG_ENABLED = Boolean(process.env.NEXT_PUBLIC_POSTHOG_KEY);

function applyPostHogPreference(enabled: boolean, userId?: string): void {
  if (!POSTHOG_ENABLED) return;

  if (enabled) {
    posthog.opt_in_capturing();
    if (userId) posthog.identify(userId);
    return;
  }

  posthog.reset();
  posthog.opt_out_capturing();
}

/** Mounts product/performance telemetry only after the signed-in user opts in. */
export function PrivacyAwareTelemetry() {
  const [enabled, setEnabled] = useState(false);

  const syncPreference = useCallback(async (knownUserId?: string) => {
    const supabase = createClient();
    const userId =
      knownUserId ?? (await supabase.auth.getUser()).data.user?.id;

    if (!userId) {
      setEnabled(false);
      applyPostHogPreference(false);
      return;
    }

    const { data: preference, error: preferenceError } = await supabase
      .from("user_privacy_settings")
      .select("analytics_opt_in")
      .eq("user_id", userId)
      .maybeSingle();

    if (preferenceError) {
      setEnabled(false);
      applyPostHogPreference(false);
      return;
    }

    let consent = null;
    if (!preference) {
      const result = await supabase
        .from("consent_records")
        .select("consent_given")
        .eq("user_id", userId)
        .eq("consent_type", "analytics_collection")
        .order("created_at", { ascending: false })
        .limit(1)
        .maybeSingle();
      if (!result.error) consent = result.data;
    }

    const nextEnabled = resolveAnalyticsOptIn(preference, consent);
    setEnabled(nextEnabled);
    applyPostHogPreference(nextEnabled, userId);
  }, []);

  useEffect(() => {
    const supabase = createClient();
    void syncPreference();

    const { data: authListener } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        queueMicrotask(() => void syncPreference(session?.user.id));
      },
    );

    function handleConsentChange() {
      void syncPreference();
    }

    window.addEventListener(
      ANALYTICS_CONSENT_CHANGED_EVENT,
      handleConsentChange,
    );

    return () => {
      authListener.subscription.unsubscribe();
      window.removeEventListener(
        ANALYTICS_CONSENT_CHANGED_EVENT,
        handleConsentChange,
      );
    };
  }, [syncPreference]);

  if (!enabled) return null;

  return (
    <>
      <Analytics />
      <SpeedInsights />
    </>
  );
}
