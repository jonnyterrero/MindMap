"use server";

import { createClient } from "@/lib/supabase-server";

export async function getLast30DaysEntries() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) return [];

  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
  const startDate = thirtyDaysAgo.toISOString().split("T")[0];

  const { data } = await supabase
    .from("mindmap_entries")
    .select("*")
    .eq("user_id", user.id)
    .gte("entry_date", startDate)
    .order("entry_date", { ascending: true });

  return data ?? [];
}

type Entry = {
  sleep_minutes: number | null;
  sleep_quality: number | null;
  anxiety: number | null;
  depression: number | null;
  mood_valence: number | null;
  migraine: boolean;
  migraine_intensity: number | null;
  migraine_aura: boolean | null;
};

/**
 * Computes a 0–100 migraine risk score based on known triggers:
 * poor sleep, high anxiety/depression, low mood, and recent migraine history.
 */
export async function getMigraineRiskToday() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) return null;

  const sevenDaysAgo = new Date();
  sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
  const startDate = sevenDaysAgo.toISOString().split("T")[0];

  const { data: entries } = await supabase
    .from("mindmap_entries")
    .select(
      "entry_date, sleep_minutes, sleep_quality, anxiety, depression, mood_valence, migraine, migraine_intensity, migraine_aura"
    )
    .eq("user_id", user.id)
    .gte("entry_date", startDate)
    .order("entry_date", { ascending: false });

  if (!entries || entries.length === 0) {
    return { score: 0, factors: [] as string[], entries: 0 };
  }

  const factors: string[] = [];
  let score = 0;

  const latest = entries[0] as Entry;

  // Sleep factors
  if (latest.sleep_minutes !== null && latest.sleep_minutes < 360) {
    score += 20;
    factors.push("Less than 6 hours of sleep");
  } else if (latest.sleep_minutes !== null && latest.sleep_minutes < 420) {
    score += 10;
    factors.push("Less than 7 hours of sleep");
  }

  if (latest.sleep_quality !== null && latest.sleep_quality <= 2) {
    score += 15;
    factors.push("Poor sleep quality");
  }

  // Stress/mood factors
  if (latest.anxiety !== null && latest.anxiety >= 7) {
    score += 15;
    factors.push("High anxiety");
  }

  if (latest.depression !== null && latest.depression >= 7) {
    score += 10;
    factors.push("High depression");
  }

  if (latest.mood_valence !== null && latest.mood_valence <= -2) {
    score += 10;
    factors.push("Low mood");
  }

  // Recent migraine history
  const recentMigraines = entries.filter(
    (e: Entry) => e.migraine === true
  ).length;
  if (recentMigraines >= 3) {
    score += 20;
    factors.push(`${recentMigraines} migraines in last 7 days`);
  } else if (recentMigraines >= 1) {
    score += 10;
    factors.push(`${recentMigraines} migraine(s) in last 7 days`);
  }

  // Recent aura
  const recentAura = entries.some(
    (e: Entry) => e.migraine_aura === true
  );
  if (recentAura) {
    score += 10;
    factors.push("Recent aura episodes");
  }

  return {
    score: Math.min(score, 100),
    factors,
    entries: entries.length,
  };
}
