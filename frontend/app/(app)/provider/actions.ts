"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";
import { computeCorrelations, type Correlation } from "@/lib/correlation-engine";

export type ProviderPermissions = {
  read_entries: boolean;
  read_reports: boolean;
  read_predictions: boolean;
};

export type PatientSummary = {
  patientId: string;
  name: string;
  latestRiskLevel: string | null;
  lastReportLabel: string | null;
  permissions: ProviderPermissions;
};

export type PatientPrediction = {
  prediction_type: string;
  risk_level: string;
  risk_score: number;
  predicted_at: string;
};
export type PatientReport = {
  id: string;
  report_type: string;
  period_start: string;
  period_end: string;
  summary_markdown: string | null;
};
export type PatientDetail = {
  name: string;
  permissions: ProviderPermissions;
  predictions: PatientPrediction[];
  reports: PatientReport[];
  correlations: Correlation[];
};

export type MyGrant = {
  id: string;
  providerName: string;
  granted_at: string;
  revoked_at: string | null;
};

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export async function getMyRole(): Promise<"patient" | "provider"> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return "patient";
  const { data } = await supabase.from("profiles").select("role").eq("id", user.id).maybeSingle();
  return (data?.role as "patient" | "provider") ?? "patient";
}

function latestPerType(rows: PatientPrediction[]): PatientPrediction[] {
  const seen = new Set<string>();
  const out: PatientPrediction[] = [];
  for (const r of rows) {
    if (seen.has(r.prediction_type)) continue;
    seen.add(r.prediction_type);
    out.push(r);
  }
  return out;
}

/** Provider: list patients who granted access (RLS scopes the grant read). */
export async function getMyPatients(): Promise<PatientSummary[]> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];

  const { data: grants } = await supabase
    .from("mindmap_provider_access")
    .select("patient_user_id, permissions")
    .eq("provider_user_id", user.id)
    .is("revoked_at", null);
  if (!grants || grants.length === 0) return [];

  const ids = grants.map((g) => g.patient_user_id as string);
  // These reads succeed via the provider-read RLS policies (017).
  const [profilesRes, predsRes, reportsRes] = await Promise.all([
    supabase.from("profiles").select("id, display_name").in("id", ids),
    supabase.from("mindmap_predictions").select("user_id, risk_level, predicted_at").in("user_id", ids).order("predicted_at", { ascending: false }),
    supabase.from("mindmap_ai_reports").select("user_id, report_type, period_end").in("user_id", ids).order("period_end", { ascending: false }),
  ]);

  const nameById = new Map((profilesRes.data ?? []).map((p) => [p.id as string, p.display_name as string | null]));
  const riskById = new Map<string, string>();
  for (const r of predsRes.data ?? []) if (!riskById.has(r.user_id as string)) riskById.set(r.user_id as string, r.risk_level as string);
  const reportById = new Map<string, string>();
  for (const r of reportsRes.data ?? []) if (!reportById.has(r.user_id as string)) reportById.set(r.user_id as string, `${r.report_type} · ${r.period_end}`);

  return grants.map((g) => ({
    patientId: g.patient_user_id as string,
    name: nameById.get(g.patient_user_id as string) || "Patient",
    latestRiskLevel: riskById.get(g.patient_user_id as string) ?? null,
    lastReportLabel: reportById.get(g.patient_user_id as string) ?? null,
    permissions: g.permissions as ProviderPermissions,
  }));
}

/** Provider: detailed read-only summary for one patient, gated by grant + permissions. */
export async function getPatientSummary(
  patientUserId: string,
): Promise<{ error: string } | PatientDetail> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { data: grant } = await supabase
    .from("mindmap_provider_access")
    .select("permissions")
    .eq("provider_user_id", user.id)
    .eq("patient_user_id", patientUserId)
    .is("revoked_at", null)
    .maybeSingle();
  if (!grant) return { error: "You don't have access to this patient." };

  const perms = grant.permissions as ProviderPermissions;
  const profileRes = await supabase.from("profiles").select("display_name").eq("id", patientUserId).maybeSingle();

  let predictions: PatientPrediction[] = [];
  let reports: PatientReport[] = [];
  let correlations: Correlation[] = [];

  if (perms.read_predictions) {
    const { data } = await supabase
      .from("mindmap_predictions")
      .select("prediction_type, risk_level, risk_score, predicted_at")
      .eq("user_id", patientUserId)
      .order("predicted_at", { ascending: false })
      .limit(20);
    predictions = latestPerType((data as PatientPrediction[] | null) ?? []);
  }
  if (perms.read_reports) {
    const { data } = await supabase
      .from("mindmap_ai_reports")
      .select("id, report_type, period_start, period_end, summary_markdown")
      .eq("user_id", patientUserId)
      .order("period_start", { ascending: false })
      .limit(12);
    reports = (data as PatientReport[] | null) ?? [];
  }
  if (perms.read_entries) {
    const { data } = await supabase
      .from("mindmap_entries")
      .select("entry_date, sleep_minutes, sleep_quality, mood_valence, anxiety, depression, focus, productivity, migraine_intensity")
      .eq("user_id", patientUserId)
      .order("entry_date", { ascending: false })
      .limit(90);
    if (data && data.length >= 10) correlations = computeCorrelations(data as Record<string, unknown>[]);
  }

  return {
    name: (profileRes.data?.display_name as string | null) || "Patient",
    permissions: perms,
    predictions,
    reports,
    correlations,
  };
}

// ---------- Patient side: grant / revoke / list ----------

/** Patient grants a provider access by their provider code (provider's user id). */
export async function grantProviderAccess(
  providerCode: string,
): Promise<{ error: string } | { success: true; providerName: string }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const code = providerCode.trim();
  if (!UUID_RE.test(code)) return { error: "That doesn't look like a valid provider code." };
  if (code === user.id) return { error: "That's your own code." };

  const { data: prov } = await supabase.rpc("lookup_provider", { p_code: code });
  const provider = Array.isArray(prov) ? prov[0] : prov;
  if (!provider) return { error: "No provider found with that code." };

  const { error } = await supabase
    .from("mindmap_provider_access")
    .upsert(
      { patient_user_id: user.id, provider_user_id: code, revoked_at: null },
      { onConflict: "patient_user_id,provider_user_id" },
    );
  if (error) return { error: error.message };

  revalidatePath("/settings");
  return { success: true, providerName: (provider.display_name as string) || "Provider" };
}

export async function revokeProviderAccess(grantId: string) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };
  const { error } = await supabase
    .from("mindmap_provider_access")
    .update({ revoked_at: new Date().toISOString() })
    .eq("id", grantId)
    .eq("patient_user_id", user.id);
  if (error) return { error: error.message };
  revalidatePath("/settings");
  return { success: true };
}

export async function getMyGrants(): Promise<MyGrant[]> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];

  const { data: grants } = await supabase
    .from("mindmap_provider_access")
    .select("id, provider_user_id, granted_at, revoked_at")
    .eq("patient_user_id", user.id)
    .order("granted_at", { ascending: false });
  if (!grants || grants.length === 0) return [];

  // Resolve provider display names (provider can be looked up; patient cannot
  // read provider profiles via RLS, so use the SECURITY DEFINER lookup).
  const out: MyGrant[] = [];
  for (const g of grants) {
    const { data: prov } = await supabase.rpc("lookup_provider", { p_code: g.provider_user_id as string });
    const provider = Array.isArray(prov) ? prov[0] : prov;
    out.push({
      id: g.id as string,
      providerName: (provider?.display_name as string) || "Provider",
      granted_at: g.granted_at as string,
      revoked_at: (g.revoked_at as string | null) ?? null,
    });
  }
  return out;
}
