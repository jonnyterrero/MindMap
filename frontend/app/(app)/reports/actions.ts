"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";
import { buildReportForUser, type BuiltReport } from "@/lib/report-core";

export type ReportRow = BuiltReport;

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

  const result = await buildReportForUser(supabase, user.id, reportType);
  if ("error" in result) return { error: result.error };
  if ("skipped" in result) return { error: "Report already generated for this period." };

  revalidatePath("/reports");
  return { report: result.report };
}
