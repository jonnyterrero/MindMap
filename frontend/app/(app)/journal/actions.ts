"use server";

import { createClient, createServiceClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";
import { reflectOnJournalText, REFLECTION_MODEL } from "@/lib/ai-reflection";
import { detectCrisis, CRISIS_RESOURCES, type CrisisSeverity } from "@/lib/crisis-detection";
import { AnalyticsEvent } from "@/lib/analytics-events";
import { captureServerEvent } from "@/lib/analytics-server";
import { consumeAiQuota, quotaExceededMessage } from "@/lib/ai-rate-limit";
import {
  isEncryptionEnabled,
  encryptJournalBodyForUser,
  decryptJournalRow,
  decryptJournalRows,
} from "@/lib/journal-crypto";

export type CrisisFlag = { severity: CrisisSeverity; eventId: string | null };

export type JournalAnalysis = {
  journal_entry_id: string;
  summary: string | null;
  reflection_question: string | null;
  tags: string[];
};

export type JournalPayload = {
  entry_date: string;
  title: string | null;
  content: string;
  mood_tags: string[];
  is_private: boolean;
};

export async function getJournalEntries() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];

  const { data } = await supabase
    .from("mindmap_journal_entries")
    .select("*")
    .eq("user_id", user.id)
    .order("entry_date", { ascending: false })
    .order("entry_time", { ascending: false })
    .limit(50);

  const rows = data ?? [];
  if (!isEncryptionEnabled() || rows.length === 0) return rows;
  // Service-role client for the DEK lookup (mindmap_journal_user_keys is
  // service-role write-only; select scoped to the caller in the row rule).
  const admin = await createServiceClient();
  return decryptJournalRows(admin, rows);
}

export async function createJournalEntry(
  payload: JournalPayload,
): Promise<{ error: string } | { success: true; crisis: CrisisFlag | null }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  // Encryption cutover: when JOURNAL_ENCRYPTION_MASTER_KEY is set, encrypt
  // the body in-process and clear plaintext `content` before insert (the
  // journal_encryption_exclusivity CHECK enforces exactly-one). Unset =
  // legacy plaintext path; matches how rows have always been written.
  let insertRow: Record<string, unknown> = { user_id: user.id, ...payload };
  if (isEncryptionEnabled()) {
    const admin = await createServiceClient();
    const patch = await encryptJournalBodyForUser(admin, user.id, payload.content);
    insertRow = { user_id: user.id, ...payload, ...patch };
  }

  const { data: inserted, error } = await supabase
    .from("mindmap_journal_entries")
    .insert(insertRow)
    .select("id")
    .single();

  if (error) return { error: error.message };
  revalidatePath("/journal");
  await captureServerEvent(user.id, AnalyticsEvent.JournalCreated);

  // Crisis trigger point: scan the plaintext we already have in payload.
  // Reading `content` back from the row would be null under encryption.
  const severity = detectCrisis(payload.content);
  let crisis: CrisisFlag | null = null;
  if (severity) {
    const { data: ev } = await supabase
      .from("mindmap_crisis_events")
      .insert({
        user_id: user.id,
        severity,
        trigger_source: "journal_entry",
        trigger_content_ref: inserted?.id ?? null,
        resources_shown: CRISIS_RESOURCES.map((r) => r.label),
      })
      .select("id")
      .single();
    crisis = { severity, eventId: (ev?.id as string) ?? null };
  }

  return { success: true, crisis };
}

export async function updateJournalEntry(id: string, payload: Partial<JournalPayload>) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  // Only re-encrypt when the body itself is in the patch; title/mood/privacy
  // updates go through as plain column updates and preserve whatever encrypt
  // state the row already has (the exclusivity CHECK stays satisfied because
  // we're not touching either `content` or `body_encrypted`).
  let updateRow: Record<string, unknown> = { ...payload };
  if (payload.content !== undefined && isEncryptionEnabled()) {
    const admin = await createServiceClient();
    const patch = await encryptJournalBodyForUser(admin, user.id, payload.content);
    updateRow = { ...payload, ...patch };
  }

  const { error } = await supabase
    .from("mindmap_journal_entries")
    .update(updateRow)
    .eq("id", id)
    .eq("user_id", user.id);

  if (error) return { error: error.message };
  revalidatePath("/journal");
  return { success: true };
}

/** Whether this user has opted into AI journal reflection. */
export async function getAiReflectionEnabled(): Promise<boolean> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return false;

  const { data } = await supabase
    .from("profiles")
    .select("ai_reflection_enabled")
    .eq("id", user.id)
    .maybeSingle();

  return Boolean(data?.ai_reflection_enabled);
}

/** All saved AI reflections for this user, keyed by journal_entry_id on the client. */
export async function getJournalAnalyses(): Promise<JournalAnalysis[]> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];

  const { data } = await supabase
    .from("mindmap_journal_ai_analysis")
    .select("journal_entry_id, summary, reflection_question, tags")
    .eq("user_id", user.id);

  return (data as JournalAnalysis[] | null) ?? [];
}

/**
 * Generate (or regenerate) a gentle AI reflection for one journal entry and
 * persist it. Opt-in only — returns an error if the user hasn't enabled it.
 */
export async function reflectOnJournalEntry(
  entryId: string,
): Promise<{ error: string } | { analysis: JournalAnalysis }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { data: profile } = await supabase
    .from("profiles")
    .select("ai_reflection_enabled")
    .eq("id", user.id)
    .maybeSingle();
  if (!profile?.ai_reflection_enabled) {
    return { error: "Turn on AI reflection in Settings first." };
  }

  const { data: entry } = await supabase
    .from("mindmap_journal_entries")
    .select("id, content, body_encrypted, encryption_algo, encryption_key_id")
    .eq("id", entryId)
    .eq("user_id", user.id)
    .maybeSingle();
  if (!entry) return { error: "Entry not found." };

  // Decrypt in-process before shipping to Anthropic. The model DOES see the
  // plaintext (that's the point of the reflection feature) -- envelope
  // encryption protects the DB / backups, not the AI request path. Privacy
  // policy calls this out.
  let plaintext: string | null = entry.content as string | null;
  if (isEncryptionEnabled() && entry.body_encrypted) {
    const admin = await createServiceClient();
    const decrypted = await decryptJournalRow(admin, entry);
    plaintext = decrypted.content ?? null;
  }
  if (!plaintext) return { error: "Entry is empty." };

  // Daily per-user quota (cost/abuse guardrail).
  const quotaCheck = await consumeAiQuota(supabase, "journal_reflection");
  if (!quotaCheck.allowed) {
    return { error: quotaExceededMessage("journal_reflection") };
  }

  let reflection;
  try {
    reflection = await reflectOnJournalText(plaintext);
  } catch (e) {
    return { error: e instanceof Error ? e.message : "Reflection failed." };
  }

  const { error } = await supabase.from("mindmap_journal_ai_analysis").upsert(
    {
      user_id: user.id,
      journal_entry_id: entryId,
      summary: reflection.summary,
      reflection_question: reflection.reflectionQuestion,
      tags: reflection.tags,
      model: REFLECTION_MODEL,
    },
    { onConflict: "journal_entry_id" },
  );
  if (error) return { error: error.message };

  revalidatePath("/journal");
  return {
    analysis: {
      journal_entry_id: entryId,
      summary: reflection.summary,
      reflection_question: reflection.reflectionQuestion,
      tags: reflection.tags,
    },
  };
}

export async function deleteJournalEntry(id: string) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("mindmap_journal_entries")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id);

  if (error) return { error: error.message };
  revalidatePath("/journal");
  return { success: true };
}
