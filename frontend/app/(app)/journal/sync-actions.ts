"use server";

import { createClient, createServiceClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";
import { detectCrisis, CRISIS_RESOURCES } from "@/lib/crisis-detection";
import { isEncryptionEnabled, encryptJournalBodyForUser } from "@/lib/journal-crypto";
import type { JournalPayload } from "./actions";

/**
 * Flush journal entries that were created offline. Inserts are new rows, so
 * there are no update conflicts. Crisis detection still runs per entry
 * (against the plaintext we're about to encrypt, not the DB row — the row
 * won't have `content` populated once encryption is enabled).
 */
export async function syncQueuedEntries(
  payloads: JournalPayload[],
): Promise<{ error: string } | { synced: number }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };
  if (payloads.length === 0) return { synced: 0 };

  // Build the insert rows. When encryption is enabled, encrypt each body
  // in-process; getOrCreateActiveUserDek dedupes internally (same user, one
  // DEK lookup, N encrypts) so the loop is cheap.
  let insertRows: Record<string, unknown>[];
  if (isEncryptionEnabled()) {
    const admin = await createServiceClient();
    insertRows = [];
    for (const p of payloads) {
      const patch = await encryptJournalBodyForUser(admin, user.id, p.content);
      insertRows.push({ user_id: user.id, ...p, ...patch });
    }
  } else {
    insertRows = payloads.map((p) => ({ user_id: user.id, ...p }));
  }

  const { data, error } = await supabase
    .from("mindmap_journal_entries")
    .insert(insertRows)
    .select("id");
  if (error) return { error: error.message };

  // Crisis detection: use the plaintext from `payloads[i].content`, aligned
  // to the inserted-row ids returned in order. Reading `content` back from
  // the row would be null under encryption.
  const inserted = data ?? [];
  for (let i = 0; i < inserted.length; i++) {
    const plaintext = payloads[i]?.content ?? "";
    const severity = detectCrisis(plaintext);
    if (severity) {
      await supabase.from("mindmap_crisis_events").insert({
        user_id: user.id,
        severity,
        trigger_source: "journal_entry",
        trigger_content_ref: inserted[i].id,
        resources_shown: CRISIS_RESOURCES.map((r) => r.label),
      });
    }
  }

  revalidatePath("/journal");
  return { synced: inserted.length };
}
