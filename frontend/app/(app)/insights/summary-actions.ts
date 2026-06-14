"use server";

import { createClient } from "@/lib/supabase-server";

/**
 * Read-only access to the Tier-0 clinician summary written by the Python batch
 * (`mindmap_ml_summaries`). The app never computes this — it only reads the
 * latest gated, evidence-cited summary for the signed-in user (RLS-scoped).
 */

export type SummaryTrajectory = {
  metric: string;
  label: string;
  direction: string;
  mean: number;
  statement: string;
};

export type SummaryPattern = { statement: string; citations: string[] };

export type SummaryWatchItem = {
  outcome: string;
  horizon: number;
  probability: number;
  method: string;
  statement: string;
};

export type SummaryInstrument = { total: number; severity: string; disclaimer: string };

export type SummaryCrisis = {
  severity: string;
  title: string;
  body: string;
  resources: { label: string; detail: string; href?: string }[];
};

export type ClinicianSummaryPayload = {
  date_range: string[];
  abstained: boolean;
  completeness: {
    logged_days: number;
    span_days: number;
    adherence: number;
    current_streak: number;
    longest_streak: number;
  };
  readiness: {
    logged_days: number;
    recommended_min_days: number;
    ready: boolean;
    days_remaining: number;
  };
  trajectories: SummaryTrajectory[];
  detected_patterns: SummaryPattern[];
  watch_items: SummaryWatchItem[];
  instruments: { phq9?: SummaryInstrument; gad7?: SummaryInstrument };
  safety_flags: string[];
  crisis: SummaryCrisis | null;
  disclaimers: string[];
};

export type ClinicianSummaryRow = {
  id: string;
  period_start: string | null;
  period_end: string;
  abstained: boolean;
  payload: ClinicianSummaryPayload;
  model_version: string;
  created_at: string;
};

export async function getLatestClinicianSummary(): Promise<ClinicianSummaryRow | null> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return null;

  const { data } = await supabase
    .from("mindmap_ml_summaries")
    .select("id, period_start, period_end, abstained, payload, model_version, created_at")
    .eq("user_id", user.id)
    .order("period_end", { ascending: false })
    .limit(1)
    .maybeSingle();

  return (data as ClinicianSummaryRow | null) ?? null;
}
