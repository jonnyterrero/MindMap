"use server";

import { createClient } from "@/lib/supabase-server";
import { computeMigraineRisk, computeMoodTrend } from "@/lib/insights-engine";
import { computeCorrelations, type Correlation } from "@/lib/correlation-engine";
import { revalidatePath } from "next/cache";

const CORRELATION_MIN_DAYS = 10;

export async function generateInsights() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { data: entries } = await supabase
    .from("mindmap_entries")
    .select("*")
    .eq("user_id", user.id)
    .order("entry_date", { ascending: false })
    .limit(30);

  if (!entries || entries.length === 0) return { error: "No data yet" };

  const latestEntry = entries[0];
  const insights = [
    computeMigraineRisk(entries),
    computeMoodTrend(entries),
  ];

  for (const insight of insights) {
    await supabase.from("mindmap_insights").upsert(
      {
        user_id: user.id,
        entry_id: latestEntry.id,
        insight_type: insight.insight_type,
        risk_level: insight.risk_level,
        score: insight.score,
        reasons: insight.reasons,
        signals: insight.signals,
        recommendation: insight.recommendation,
        computed_at: new Date().toISOString(),
      },
      { onConflict: "user_id,entry_id,insight_type", ignoreDuplicates: false }
    );
  }

  revalidatePath("/insights");
  revalidatePath("/dashboard");

  const { data: saved } = await supabase
    .from("mindmap_insights")
    .select("*")
    .eq("user_id", user.id)
    .order("computed_at", { ascending: false })
    .limit(10);

  return saved ?? [];
}

export async function getLatestInsights() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];

  const { data } = await supabase
    .from("mindmap_insights")
    .select("*")
    .eq("user_id", user.id)
    .order("computed_at", { ascending: false })
    .limit(10);

  return data ?? [];
}

/**
 * Compute Pearson correlations across recent entries. Returns [] until the
 * user has at least CORRELATION_MIN_DAYS of data, so the section stays hidden
 * (and the UI uncluttered) until it's meaningful.
 */
export async function getCorrelations(): Promise<Correlation[]> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];

  const { data: entries } = await supabase
    .from("mindmap_entries")
    .select(
      "entry_date, sleep_minutes, sleep_quality, mood_valence, anxiety, depression, focus, productivity, migraine_intensity"
    )
    .eq("user_id", user.id)
    .order("entry_date", { ascending: false })
    .limit(90);

  if (!entries || entries.length < CORRELATION_MIN_DAYS) return [];

  // Merge weather (if the user tracks it) by entry_date so weather vs symptom
  // correlations can surface. Best-effort: absent weather just means fewer pairs.
  // Body-map pain is merged the same way (max logged intensity per entry).
  const [{ data: weather }, { data: sensations }] = await Promise.all([
    supabase
      .from("mindmap_weather_daily")
      .select("entry_date, pressure, humidity, temp_max")
      .eq("user_id", user.id)
      .limit(90),
    // RLS-scoped via entry_id; pull the parent entry_date to align with daily rows.
    supabase
      .from("mindmap_body_sensations")
      .select("intensity, mindmap_entries!inner(entry_date)")
      .limit(2000),
  ]);

  type WeatherRow = {
    entry_date: string;
    pressure: number | null;
    humidity: number | null;
    temp_max: number | null;
  };
  const weatherByDate = new Map<string, WeatherRow>(
    (weather as WeatherRow[] | null ?? []).map((w) => [w.entry_date, w]),
  );

  // Max body-pain intensity per entry_date (a day's worst logged sensation).
  type SensationJoin = {
    intensity: number | null;
    mindmap_entries: { entry_date: string } | { entry_date: string }[] | null;
  };
  const bodyPainByDate = new Map<string, number>();
  for (const s of (sensations as SensationJoin[] | null) ?? []) {
    const rel = Array.isArray(s.mindmap_entries) ? s.mindmap_entries[0] : s.mindmap_entries;
    const date = rel?.entry_date;
    if (!date || typeof s.intensity !== "number") continue;
    const prev = bodyPainByDate.get(date);
    if (prev == null || s.intensity > prev) bodyPainByDate.set(date, s.intensity);
  }

  const merged = (entries as Record<string, unknown>[]).map((e) => {
    const date = e.entry_date as string;
    const w = weatherByDate.get(date);
    const bodyPain = bodyPainByDate.get(date);
    return {
      ...e,
      ...(w ? { pressure: w.pressure, humidity: w.humidity, temp_max: w.temp_max } : {}),
      ...(bodyPain != null ? { body_pain: bodyPain } : {}),
    };
  });

  return computeCorrelations(merged);
}

export async function getInsightHistory(insightType: string) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];

  const { data } = await supabase
    .from("mindmap_insights")
    .select("score, risk_level, computed_at, reasons")
    .eq("user_id", user.id)
    .eq("insight_type", insightType)
    .order("computed_at", { ascending: false })
    .limit(30);

  return data ?? [];
}
