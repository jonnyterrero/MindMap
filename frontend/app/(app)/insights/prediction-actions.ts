"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";
import {
  computePredictions,
  type WearableLatest,
  type WeatherSignal,
  type BodyPainSignal,
} from "@/lib/prediction-engine";

export type PredictionRow = {
  id: string;
  prediction_type: string;
  risk_score: number;
  risk_level: string;
  confidence: number;
  contributing_factors: { factor: string; weight: number; detail?: string }[];
  predicted_at: string;
  acknowledged_at: string | null;
  outcome_recorded: string | null;
};

const DEDUP_WINDOW_MS = 12 * 60 * 60 * 1000;

/**
 * Gather last-14d signals, run the prediction engine, and upsert one row per
 * type (re-using a row from the last 12h to avoid spamming). Returns the rows.
 */
export async function runPredictionEngine(): Promise<{ error: string } | { predictions: PredictionRow[] }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const since = new Date();
  since.setDate(since.getDate() - 14);
  const sinceISO = since.toISOString().split("T")[0];

  const [entriesRes, wearableRes, weatherRes, sensationsRes] = await Promise.all([
    supabase
      .from("mindmap_entries")
      .select(
        "entry_date, sleep_minutes, sleep_quality, anxiety, depression, mood_valence, focus, productivity, migraine, migraine_intensity",
      )
      .eq("user_id", user.id)
      .gte("entry_date", sinceISO)
      .order("entry_date", { ascending: false }),
    supabase
      .from("mindmap_wearable_data")
      .select("metric_type, value, recorded_at")
      .eq("user_id", user.id)
      .in("metric_type", ["hrv", "sleep_score", "resting_hr"])
      .order("recorded_at", { ascending: false })
      .limit(60),
    supabase
      .from("mindmap_weather_daily")
      .select("pressure_change, pollen_level")
      .eq("user_id", user.id)
      .order("entry_date", { ascending: false })
      .limit(1)
      .maybeSingle(),
    // Body-map sensations are RLS-scoped via entry_id → mindmap_entries.user_id.
    supabase
      .from("mindmap_body_sensations")
      .select("intensity, created_at")
      .gte("created_at", sinceISO)
      .order("created_at", { ascending: false })
      .limit(500),
  ]);

  const entries = entriesRes.data ?? [];
  if (entries.length === 0) return { predictions: [] };

  // Latest value per wearable metric.
  const wearable: WearableLatest = {};
  for (const row of wearableRes.data ?? []) {
    const k = row.metric_type as keyof WearableLatest;
    if (wearable[k] == null) wearable[k] = row.value as number;
  }
  const weather: WeatherSignal = {
    pressure_change: (weatherRes.data?.pressure_change as number | null) ?? null,
    pollen_level: (weatherRes.data?.pollen_level as string | null) ?? null,
  };

  // Aggregate logged body pain: mean severity over the window + days-with-pain
  // in the last 7 days (drives the pain-flare score and its confidence).
  const sensRows = (sensationsRes.data ?? []) as { intensity: number; created_at: string }[];
  const sevenAgoMs = Date.now() - 7 * 24 * 60 * 60 * 1000;
  const intensities = sensRows
    .map((r) => r.intensity)
    .filter((n): n is number => typeof n === "number");
  const recentDays = new Set<string>();
  for (const r of sensRows) {
    if (new Date(r.created_at).getTime() >= sevenAgoMs) {
      recentDays.add(r.created_at.split("T")[0]);
    }
  }
  const bodyPain: BodyPainSignal = {
    avgIntensity: intensities.length
      ? intensities.reduce((a, b) => a + b, 0) / intensities.length
      : null,
    daysWithPain: recentDays.size,
  };

  const computed = computePredictions({ entries, wearable, weather, bodyPain });
  const nowMs = Date.now();
  const out: PredictionRow[] = [];

  for (const p of computed) {
    // Re-use a row from the last 12h for the same type, else insert.
    const { data: existing } = await supabase
      .from("mindmap_predictions")
      .select("id, predicted_at")
      .eq("user_id", user.id)
      .eq("prediction_type", p.prediction_type)
      .order("predicted_at", { ascending: false })
      .limit(1)
      .maybeSingle();

    const fresh =
      existing && nowMs - new Date(existing.predicted_at as string).getTime() < DEDUP_WINDOW_MS;

    const values = {
      risk_score: p.risk_score,
      risk_level: p.risk_level,
      confidence: p.confidence,
      contributing_factors: p.contributing_factors,
      model_version: p.model_version,
      predicted_at: new Date().toISOString(),
      acknowledged_at: null,
      outcome_recorded: null,
    };

    if (fresh && existing) {
      const { data } = await supabase
        .from("mindmap_predictions")
        .update(values)
        .eq("id", existing.id)
        .select("id, prediction_type, risk_score, risk_level, confidence, contributing_factors, predicted_at, acknowledged_at, outcome_recorded")
        .single();
      if (data) out.push(data as PredictionRow);
    } else {
      const { data } = await supabase
        .from("mindmap_predictions")
        .insert({ user_id: user.id, prediction_type: p.prediction_type, ...values })
        .select("id, prediction_type, risk_score, risk_level, confidence, contributing_factors, predicted_at, acknowledged_at, outcome_recorded")
        .single();
      if (data) out.push(data as PredictionRow);
    }
  }

  revalidatePath("/insights");
  revalidatePath("/home");
  return { predictions: out };
}

/** Latest prediction per type for display. */
export async function getLatestPredictions(): Promise<PredictionRow[]> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];

  const { data } = await supabase
    .from("mindmap_predictions")
    .select("id, prediction_type, risk_score, risk_level, confidence, contributing_factors, predicted_at, acknowledged_at, outcome_recorded")
    .eq("user_id", user.id)
    .order("predicted_at", { ascending: false })
    .limit(20);

  const seen = new Set<string>();
  const latest: PredictionRow[] = [];
  for (const row of (data as PredictionRow[] | null) ?? []) {
    if (seen.has(row.prediction_type)) continue;
    seen.add(row.prediction_type);
    latest.push(row);
  }
  return latest;
}

export async function recordPredictionOutcome(id: string, outcome: "accurate" | "inaccurate") {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("mindmap_predictions")
    .update({ outcome_recorded: outcome, acknowledged_at: new Date().toISOString() })
    .eq("id", id)
    .eq("user_id", user.id);
  if (error) return { error: error.message };
  revalidatePath("/insights");
  return { success: true };
}
