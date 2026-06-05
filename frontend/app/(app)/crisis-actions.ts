"use server";

import { createClient } from "@/lib/supabase-server";
import { CRISIS_RESOURCES, type CrisisSeverity } from "@/lib/crisis-detection";

/** Log a crisis event when concerning language is detected. */
export async function logCrisisEvent(input: {
  severity: CrisisSeverity;
  source: "ai_message" | "journal_entry" | "voice" | "manual";
  contentRef?: string | null;
}): Promise<{ error: string } | { id: string }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { data, error } = await supabase
    .from("mindmap_crisis_events")
    .insert({
      user_id: user.id,
      severity: input.severity,
      trigger_source: input.source,
      trigger_content_ref: input.contentRef ?? null,
      resources_shown: CRISIS_RESOURCES.map((r) => r.label),
    })
    .select("id")
    .single();

  if (error) return { error: error.message };
  return { id: data.id as string };
}

export async function acknowledgeCrisisEvent(id: string) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("mindmap_crisis_events")
    .update({ acknowledged_at: new Date().toISOString() })
    .eq("id", id)
    .eq("user_id", user.id);
  if (error) return { error: error.message };
  return { success: true };
}
