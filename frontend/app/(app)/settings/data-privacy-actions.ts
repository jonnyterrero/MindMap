"use server";

import { createClient, createServiceClient } from "@/lib/supabase-server";
import { isEncryptionEnabled, decryptJournalRows } from "@/lib/journal-crypto";

// User-owned tables exported via the caller's own RLS session (they can read
// all of their own rows). `profiles` is keyed by id; the rest by user_id.
const EXPORT_TABLES = [
  "mindmap_entries",
  "mindmap_journal_entries",
  "mindmap_goals",
  "mindmap_therapy_sessions",
  "mindmap_triggers",
  "mindmap_reminders",
  "mindmap_medication_schedule",
  "mindmap_insights",
  "mindmap_predictions",
  "mindmap_weather_daily",
  "mindmap_ai_conversations",
  "mindmap_ai_reports",
  "mindmap_crisis_events",
  "mindmap_wearable_sources",
  "mindmap_wearable_data",
  "mindmap_voice_notes",
  "consent_records",
  "user_privacy_settings",
];

export type ExportBundle = {
  exportedAt: string;
  userId: string;
  tables: Record<string, unknown[]>;
};

/** Assemble all of the caller's own data into a JSON bundle (RLS-scoped). */
export async function exportUserData(): Promise<{ error: string } | { bundle: ExportBundle }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const tables: Record<string, unknown[]> = {};

  const profileRes = await supabase.from("profiles").select("*").eq("id", user.id);
  tables["profiles"] = profileRes.data ?? [];

  for (const t of EXPORT_TABLES) {
    const { data, error } = await supabase.from(t).select("*").eq("user_id", user.id);
    tables[t] = error ? [] : data ?? [];
  }

  // GDPR/CCPA: decrypt journal bodies so the bundle the user downloads is
  // plaintext they can actually read. Null out the ciphertext columns so we
  // don't ship both formats and confuse the export.
  if (isEncryptionEnabled()) {
    const journalRows = tables["mindmap_journal_entries"] ?? [];
    if (journalRows.length > 0) {
      const admin = await createServiceClient();
      const decrypted = await decryptJournalRows(
        admin,
        journalRows as Array<Record<string, unknown>>,
      );
      tables["mindmap_journal_entries"] = decrypted.map((row) => ({
        ...row,
        body_encrypted: null,
        encryption_key_id: null,
        encryption_algo: "none",
        encrypted_at: null,
      }));
    }
  }

  return {
    bundle: { exportedAt: new Date().toISOString(), userId: user.id, tables },
  };
}

/**
 * Hard-delete the caller's account. PHI child tables cascade via FKs
 * (migration 026). The data_deletion_requests audit row survives with
 * user_id nulled and retained_metadata.deleted_user_id set. Requires the
 * typed "DELETE" confirmation. Irreversible.
 */
export async function deleteAccount(
  confirmation: string,
): Promise<{ error: string } | { success: true }> {
  if (confirmation !== "DELETE") {
    return { error: 'Type "DELETE" to confirm.' };
  }

  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    return { error: "Account deletion isn't configured on the server (missing service role key)." };
  }

  const admin = await createServiceClient();

  const { data: request, error: requestErr } = await admin
    .from("data_deletion_requests")
    .insert({
      user_id: user.id,
      scope: "all",
      status: "processing",
      reason: "User-initiated account deletion",
      retained_metadata: { deleted_user_id: user.id },
    })
    .select("id")
    .single();
  if (requestErr) return { error: requestErr.message };

  // Best-effort storage cleanup (buckets may not exist yet).
  try {
    await admin.storage.from("voice-notes").remove([`${user.id}/`]);
  } catch {
    // ignore
  }

  const { error } = await admin.auth.admin.deleteUser(user.id);
  if (error) {
    await admin
      .from("data_deletion_requests")
      .update({
        status: "cancelled",
        failure_reason: error.message,
      })
      .eq("id", request.id);
    return { error: error.message };
  }

  await admin
    .from("data_deletion_requests")
    .update({
      status: "completed",
      processed_at: new Date().toISOString(),
      completed_at: new Date().toISOString(),
      processed_by: "deleteAccount",
    })
    .eq("id", request.id);

  await supabase.auth.signOut();
  return { success: true };
}
