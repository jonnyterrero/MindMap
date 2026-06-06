"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";
import { METRIC_META, type WearableMetric, type WearableSourceType } from "@/lib/wearable";

export type WearableSourceRow = {
  id: string;
  source_type: string;
  is_active: boolean;
  last_sync_at: string | null;
};
export type RecentMetric = {
  metric_type: string;
  value: number;
  unit: string | null;
  recorded_at: string;
};

export async function getWearableSources(): Promise<WearableSourceRow[]> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];
  const { data } = await supabase
    .from("mindmap_wearable_sources")
    .select("id, source_type, is_active, last_sync_at")
    .eq("user_id", user.id)
    .order("connected_at", { ascending: false });
  return (data as WearableSourceRow[] | null) ?? [];
}

/** Latest value per metric (for display + the prediction engine). */
export async function getRecentMetrics(): Promise<RecentMetric[]> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];
  const { data } = await supabase
    .from("mindmap_wearable_data")
    .select("metric_type, value, unit, recorded_at")
    .eq("user_id", user.id)
    .order("recorded_at", { ascending: false })
    .limit(60);

  const seen = new Set<string>();
  const out: RecentMetric[] = [];
  for (const r of (data as RecentMetric[] | null) ?? []) {
    if (seen.has(r.metric_type)) continue;
    seen.add(r.metric_type);
    out.push(r);
  }
  return out;
}

/** Manual metric entry (web). Stored with source_id null. */
export async function logMetric(metric: WearableMetric, value: number) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };
  if (!Number.isFinite(value)) return { error: "Enter a number." };

  const { error } = await supabase.from("mindmap_wearable_data").insert({
    user_id: user.id,
    source_id: null,
    metric_type: metric,
    value,
    unit: METRIC_META[metric]?.unit ?? null,
    recorded_at: new Date().toISOString(),
  });
  if (error) return { error: error.message };
  revalidatePath("/settings");
  return { success: true };
}

/** Register a source. Native sources (Apple Health/Health Connect) still need
 *  the mobile app for actual sync; this records the connection intent. */
export async function connectSource(sourceType: WearableSourceType) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("mindmap_wearable_sources")
    .upsert(
      { user_id: user.id, source_type: sourceType, is_active: true },
      { onConflict: "user_id,source_type" },
    );
  if (error) return { error: error.message };
  revalidatePath("/settings");
  return { success: true };
}

export async function disconnectSource(id: string) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };
  const { error } = await supabase
    .from("mindmap_wearable_sources")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id);
  if (error) return { error: error.message };
  revalidatePath("/settings");
  return { success: true };
}
