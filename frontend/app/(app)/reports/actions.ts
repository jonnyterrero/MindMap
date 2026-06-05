"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";
import { computeCorrelations } from "@/lib/correlation-engine";
import { generateReportMarkdown, type ReportContext } from "@/lib/report-generator";

export type ReportRow = {
  id: string;
  report_type: string;
  period_start: string;
  period_end: string;
  summary_markdown: string | null;
  key_insights: string[];
  generated_at: string;
};

function avg(nums: (number | null)[]): number | null {
  const v = nums.filter((n): n is number => typeof n === "number");
  return v.length ? Math.round((v.reduce((a, b) => a + b, 0) / v.length) * 10) / 10 : null;
}
function iso(d: Date) {
  return d.toISOString().split("T")[0];
}

export async function getReports(): Promise<ReportRow[]> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];
  const { data } = await supabase
    .from("mindmap_ai_reports")
    .select("id, report_type, period_start, period_end, summary_markdown, key_insights, generated_at")
    .eq("user_id", user.id)
    .order("period_start", { ascending: false })
    .limit(24);
  return (data as ReportRow[] | null) ?? [];
}

export async function generateReport(
  reportType: "weekly" | "monthly",
): Promise<{ error: string } | { report: ReportRow }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - (reportType === "weekly" ? 6 : 29));
  const periodStart = iso(start);
  const periodEnd = iso(end);

  const [entriesRes, predsRes, recentRes] = await Promise.all([
    supabase
      .from("mindmap_entries")
      .select("entry_date, sleep_minutes, mood_valence, anxiety, depression, focus, migraine, migraine_intensity")
      .eq("user_id", user.id)
      .gte("entry_date", periodStart)
      .lte("entry_date", periodEnd),
    supabase
      .from("mindmap_predictions")
      .select("outcome_recorded")
      .eq("user_id", user.id)
      .gte("predicted_at", start.toISOString()),
    supabase
      .from("mindmap_entries")
      .select("sleep_minutes, sleep_quality, mood_valence, anxiety, depression, focus, productivity, migraine_intensity")
      .eq("user_id", user.id)
      .order("entry_date", { ascending: false })
      .limit(90),
  ]);

  const entries = entriesRes.data ?? [];
  if (entries.length === 0) {
    return { error: "No check-ins in this period yet. Log a few days first." };
  }

  const stats = {
    entryCount: entries.length,
    avgSleepHours: avg(entries.map((e) => (e.sleep_minutes != null ? (e.sleep_minutes as number) / 60 : null))),
    avgMood: avg(entries.map((e) => e.mood_valence as number | null)),
    avgAnxiety: avg(entries.map((e) => e.anxiety as number | null)),
    avgDepression: avg(entries.map((e) => e.depression as number | null)),
    avgFocus: avg(entries.map((e) => e.focus as number | null)),
    migraineDays: entries.filter((e) => e.migraine === true).length,
  };

  const preds = predsRes.data ?? [];
  const predictionAccuracy = {
    total: preds.filter((p) => p.outcome_recorded != null).length,
    accurate: preds.filter((p) => p.outcome_recorded === "accurate").length,
    inaccurate: preds.filter((p) => p.outcome_recorded === "inaccurate").length,
  };

  const correlations = computeCorrelations((recentRes.data as Record<string, unknown>[]) ?? []);
  const topCorrelations = correlations.slice(0, 5).map((c) => c.statement);

  const ctx: ReportContext = {
    reportType,
    periodStart,
    periodEnd,
    stats,
    topCorrelations,
    predictionAccuracy,
  };

  let generated;
  try {
    generated = await generateReportMarkdown(ctx);
  } catch (e) {
    return { error: e instanceof Error ? e.message : "Report generation failed." };
  }

  const { data, error } = await supabase
    .from("mindmap_ai_reports")
    .upsert(
      {
        user_id: user.id,
        report_type: reportType,
        period_start: periodStart,
        period_end: periodEnd,
        summary_markdown: generated.summaryMarkdown,
        key_insights: generated.keyInsights,
        trend_data: stats,
        generated_at: new Date().toISOString(),
      },
      { onConflict: "user_id,report_type,period_start" },
    )
    .select("id, report_type, period_start, period_end, summary_markdown, key_insights, generated_at")
    .single();

  if (error) return { error: error.message };
  revalidatePath("/reports");
  return { report: data as ReportRow };
}
