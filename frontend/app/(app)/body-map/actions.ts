"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";

export type BodySensationRow = {
  id: string;
  body_part: string;
  sensation: string | null;
  intensity: number;
  notes: string | null;
  created_at: string;
  entry_id: string;
};

/** Log a body sensation against today's entry (creating the entry if needed). */
export async function logBodySensation(input: {
  bodyPart: string;
  sensation: string;
  intensity: number;
  notes?: string | null;
}): Promise<{ error: string } | { success: true }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };
  if (!input.bodyPart) return { error: "Pick a body area." };

  const today = new Date().toISOString().split("T")[0];
  let { data: entry } = await supabase
    .from("mindmap_entries")
    .select("id")
    .eq("user_id", user.id)
    .eq("entry_date", today)
    .maybeSingle();

  if (!entry) {
    const { data: created, error: e } = await supabase
      .from("mindmap_entries")
      .insert({ user_id: user.id, entry_date: today })
      .select("id")
      .single();
    if (e) return { error: e.message };
    entry = created;
  }

  const { error } = await supabase.from("mindmap_body_sensations").insert({
    entry_id: entry!.id,
    body_part: input.bodyPart,
    sensation: input.sensation,
    intensity: Math.max(0, Math.min(10, Math.round(input.intensity))),
    notes: input.notes?.trim() || null,
  });
  if (error) return { error: error.message };

  // Body pain feeds correlations + the pain-flare prediction.
  revalidatePath("/body-map");
  revalidatePath("/insights");
  revalidatePath("/today");
  return { success: true };
}

export async function removeBodySensation(id: string) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  // Defense-in-depth (in addition to RLS): confirm this sensation's parent
  // entry belongs to the caller before deleting. mindmap_body_sensations has no
  // user_id column, so ownership is verified via the entry FK.
  const { data: owned } = await supabase
    .from("mindmap_body_sensations")
    .select("id, mindmap_entries!inner(user_id)")
    .eq("id", id)
    .eq("mindmap_entries.user_id", user.id)
    .maybeSingle();
  if (!owned) return { error: "Not found" };

  const { error } = await supabase.from("mindmap_body_sensations").delete().eq("id", id);
  if (error) return { error: error.message };
  revalidatePath("/body-map");
  return { success: true };
}
