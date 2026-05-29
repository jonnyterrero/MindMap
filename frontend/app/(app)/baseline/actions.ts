"use server";

import { createClient } from "@/lib/supabase-server";
import { BASELINE_DAYS } from "@/lib/guided-plan";

function avg(values: (number | null | undefined)[]): number | null {
  const nums = values.filter((v): v is number => typeof v === "number");
  if (nums.length === 0) return null;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

export interface BaselineData {
  unlocked: boolean;
  checkInsCompleted: number;
  remaining: number;
  avgSleepHours: number | null;
  avgSleepQuality: number | null;
  migraineDays: number;
  avgMigraineIntensity: number | null;
  avgAnxiety: number | null;
  avgFocus: number | null;
  routineCompletionPct: number | null;
  medicationConsistencyPct: number | null;
  /** A single cautious, non-diagnostic observation (or null if not enough data). */
  pattern: string | null;
}

export async function getBaselineData(): Promise<BaselineData> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  const empty: BaselineData = {
    unlocked: false,
    checkInsCompleted: 0,
    remaining: BASELINE_DAYS,
    avgSleepHours: null,
    avgSleepQuality: null,
    migraineDays: 0,
    avgMigraineIntensity: null,
    avgAnxiety: null,
    avgFocus: null,
    routineCompletionPct: null,
    medicationConsistencyPct: null,
    pattern: null,
  };
  if (!user) return empty;

  const { data: entries } = await supabase
    .from("mindmap_entries")
    .select("id, entry_date, sleep_minutes, sleep_quality, migraine, migraine_intensity, anxiety, focus")
    .eq("user_id", user.id)
    .order("entry_date", { ascending: false })
    .limit(30);

  const rows = entries ?? [];
  const checkInsCompleted = rows.length;

  if (checkInsCompleted < BASELINE_DAYS) {
    return { ...empty, checkInsCompleted, remaining: BASELINE_DAYS - checkInsCompleted };
  }

  const entryIds = rows.map((r) => r.id as string);
  const totalDays = rows.length;

  const [routinesRes, medsRes] = await Promise.all([
    supabase
      .from("mindmap_entry_routines")
      .select("entry_id")
      .in("entry_id", entryIds)
      .eq("completed", true),
    supabase
      .from("mindmap_medication_adherence")
      .select("entry_id")
      .in("entry_id", entryIds)
      .eq("was_taken", true),
  ]);

  const routineDays = new Set((routinesRes.data ?? []).map((r) => r.entry_id)).size;
  const medDays = new Set((medsRes.data ?? []).map((r) => r.entry_id)).size;

  const sleepHours = rows.map((r) =>
    r.sleep_minutes != null ? (r.sleep_minutes as number) / 60 : null,
  );
  const avgSleepHours = avg(sleepHours);
  const migraineIntensities = rows
    .filter((r) => r.migraine === true)
    .map((r) => r.migraine_intensity as number | null);

  // Cautious pattern: compare migraine intensity on low-sleep vs other days.
  let pattern: string | null = null;
  const lowSleep = rows.filter((r) => r.sleep_minutes != null && (r.sleep_minutes as number) < 360);
  const otherSleep = rows.filter((r) => r.sleep_minutes != null && (r.sleep_minutes as number) >= 360);
  const lowSleepMig = avg(lowSleep.map((r) => (r.migraine ? (r.migraine_intensity as number) : null)));
  const otherMig = avg(otherSleep.map((r) => (r.migraine ? (r.migraine_intensity as number) : null)));
  if (lowSleepMig != null && otherMig != null && lowSleepMig > otherMig + 0.5) {
    pattern =
      "During your first days, lower sleep duration appeared alongside higher migraine intensity. This is a possible pattern, not a diagnosis or medical conclusion.";
  }

  return {
    unlocked: true,
    checkInsCompleted,
    remaining: 0,
    avgSleepHours,
    avgSleepQuality: avg(rows.map((r) => r.sleep_quality as number | null)),
    migraineDays: rows.filter((r) => r.migraine === true).length,
    avgMigraineIntensity: avg(migraineIntensities),
    avgAnxiety: avg(rows.map((r) => r.anxiety as number | null)),
    avgFocus: avg(rows.map((r) => r.focus as number | null)),
    routineCompletionPct: totalDays > 0 ? Math.round((routineDays / totalDays) * 100) : null,
    medicationConsistencyPct: totalDays > 0 ? Math.round((medDays / totalDays) * 100) : null,
    pattern,
  };
}
