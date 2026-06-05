"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";
import { reflectOnJournalText, REFLECTION_MODEL } from "@/lib/ai-reflection";
import { detectCrisis, CRISIS_RESOURCES, type CrisisSeverity } from "@/lib/crisis-detection";

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

  return data ?? [];
}

export async function createJournalEntry(
  payload: JournalPayload,
): Promise<{ error: string } | { success: true; crisis: CrisisFlag | null }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { data: inserted, error } = await supabase
    .from("mindmap_journal_entries")
    .insert({ user_id: user.id, ...payload })
    .select("id")
    .single();

  if (error) return { error: error.message };
  revalidatePath("/journal");

  // Crisis trigger point: scan the saved content.
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

  const { error } = await supabase
    .from("mindmap_journal_entries")
    .update(payload)
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
    .select("id, content")
    .eq("id", entryId)
    .eq("user_id", user.id)
    .maybeSingle();
  if (!entry?.content) {
    return { error: "Entry not found or empty." };
  }

  let reflection;
  try {
    reflection = await reflectOnJournalText(entry.content as string);
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
