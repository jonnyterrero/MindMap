"use server";

import { createClient } from "@/lib/supabase-server";

function isoDay(d: Date): string {
  return d.toISOString().split("T")[0];
}

/** Consecutive-day streak ending today (or yesterday, as a one-day grace). */
function computeStreak(dates: string[]): number {
  const set = new Set(dates);
  const today = new Date();
  let cursor = new Date(today);

  // If today isn't logged yet, an ongoing streak shouldn't read as broken
  // until a full day is missed — start counting from yesterday.
  if (!set.has(isoDay(cursor))) {
    cursor.setDate(cursor.getDate() - 1);
  }

  let streak = 0;
  while (set.has(isoDay(cursor))) {
    streak += 1;
    cursor.setDate(cursor.getDate() - 1);
  }
  return streak;
}

export interface HomeInsight {
  insight_type: string | null;
  risk_level: string | null;
  recommendation: string | null;
  summary: string | null;
}

export interface HomeData {
  todayScore: number | null;
  todayDone: boolean;
  checkInsCompleted: number;
  streak: number;
  latestInsight: HomeInsight | null;
}

export async function getHomeData(): Promise<HomeData> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { todayScore: null, todayDone: false, checkInsCompleted: 0, streak: 0, latestInsight: null };
  }

  const today = isoDay(new Date());

  const [entriesRes, countRes, insightRes] = await Promise.all([
    supabase
      .from("mindmap_entries")
      .select("entry_date, mindmap_score")
      .eq("user_id", user.id)
      .order("entry_date", { ascending: false })
      .limit(90),
    supabase
      .from("mindmap_entries")
      .select("id", { count: "exact", head: true })
      .eq("user_id", user.id),
    supabase
      .from("mindmap_insights")
      .select("insight_type, risk_level, recommendation, summary")
      .eq("user_id", user.id)
      .order("computed_at", { ascending: false })
      .limit(1)
      .maybeSingle(),
  ]);

  const entries = entriesRes.data ?? [];
  const todayRow = entries.find((e) => e.entry_date === today);

  return {
    todayScore: (todayRow?.mindmap_score as number | null) ?? null,
    todayDone: Boolean(todayRow),
    checkInsCompleted: countRes.count ?? 0,
    streak: computeStreak(entries.map((e) => e.entry_date as string)),
    latestInsight: (insightRes.data as HomeInsight | null) ?? null,
  };
}
