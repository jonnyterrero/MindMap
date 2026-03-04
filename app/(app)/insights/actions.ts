"use server";

import { createClient } from "@/lib/supabase-server";
import { computeMigraineRisk, computeMoodTrend } from "@/lib/insights-engine";
import { revalidatePath } from "next/cache";

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
  return { success: true, insights };
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
