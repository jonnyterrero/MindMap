"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";
import { detectCrisis, CRISIS_RESOURCES } from "@/lib/crisis-detection";
import type { JournalPayload } from "./actions";

/**
 * Flush journal entries that were created offline. Inserts are new rows, so
 * there are no update conflicts. Crisis detection still runs per entry.
 */
export async function syncQueuedEntries(
  payloads: JournalPayload[],
): Promise<{ error: string } | { synced: number }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };
  if (payloads.length === 0) return { synced: 0 };

  const { data, error } = await supabase
    .from("mindmap_journal_entries")
    .insert(payloads.map((p) => ({ user_id: user.id, ...p })))
    .select("id, content");
  if (error) return { error: error.message };

  // Crisis detection on each synced entry (best-effort logging).
  for (const row of data ?? []) {
    const severity = detectCrisis(row.content as string);
    if (severity) {
      await supabase.from("mindmap_crisis_events").insert({
        user_id: user.id,
        severity,
        trigger_source: "journal_entry",
        trigger_content_ref: row.id,
        resources_shown: CRISIS_RESOURCES.map((r) => r.label),
      });
    }
  }

  revalidatePath("/journal");
  return { synced: data?.length ?? 0 };
}
