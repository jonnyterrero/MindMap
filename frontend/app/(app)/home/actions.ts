"use server";

import { createClient } from "@/lib/supabase-server";
import { computeStreak } from "@/lib/local-date";
import { userCalendarDate } from "@/lib/user-local-today";

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

  const today = await userCalendarDate(supabase, user.id);

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
    streak: computeStreak(entries.map((e) => e.entry_date as string), today),
    latestInsight: (insightRes.data as HomeInsight | null) ?? null,
  };
}
