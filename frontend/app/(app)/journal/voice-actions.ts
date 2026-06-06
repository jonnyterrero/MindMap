"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";
import { detectCrisis, CRISIS_RESOURCES, type CrisisSeverity } from "@/lib/crisis-detection";
import { analyzeVoiceTranscript } from "@/lib/voice-sentiment";

export type VoiceSaveResult =
  | { error: string }
  | { success: true; crisis: { severity: CrisisSeverity; eventId: string | null } | null };

/**
 * Persist a transcribed voice note: creates a linked journal entry, stores the
 * voice note, runs sentiment/themes (best-effort) and crisis detection.
 */
export async function saveVoiceNote(
  transcript: string,
  durationSeconds: number,
): Promise<VoiceSaveResult> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const text = transcript.trim();
  if (!text) return { error: "Nothing was transcribed." };

  const today = new Date().toISOString().split("T")[0];

  // 1. Journal entry from the transcript.
  const { data: entry, error: entryErr } = await supabase
    .from("mindmap_journal_entries")
    .insert({
      user_id: user.id,
      entry_date: today,
      title: "Voice note",
      content: text,
      mood_tags: [],
      is_private: true,
    })
    .select("id")
    .single();
  if (entryErr) return { error: entryErr.message };
  const entryId = entry.id as string;

  // 2. Crisis detection on the transcript.
  const severity = detectCrisis(text);
  let crisis: { severity: CrisisSeverity; eventId: string | null } | null = null;
  if (severity) {
    const { data: ev } = await supabase
      .from("mindmap_crisis_events")
      .insert({
        user_id: user.id,
        severity,
        trigger_source: "voice",
        trigger_content_ref: entryId,
        resources_shown: CRISIS_RESOURCES.map((r) => r.label),
      })
      .select("id")
      .single();
    crisis = { severity, eventId: (ev?.id as string) ?? null };
  }

  // 3. Sentiment + themes (best-effort).
  let sentiment: number | null = null;
  let themes: string[] = [];
  try {
    const a = await analyzeVoiceTranscript(text);
    sentiment = a.sentiment;
    themes = a.themes;
  } catch {
    // analysis is optional
  }

  // 4. Voice note row (no audio file stored — Web Speech transcribes live).
  await supabase.from("mindmap_voice_notes").insert({
    user_id: user.id,
    entry_id: entryId,
    storage_path: `webspeech://${Date.now()}`,
    duration_seconds: Math.round(durationSeconds),
    transcript: text,
    transcript_status: "complete",
    sentiment_score: sentiment,
    themes,
  });

  // 5. Tag the journal entry with detected themes.
  if (themes.length > 0) {
    await supabase.from("mindmap_journal_entries").update({ mood_tags: themes }).eq("id", entryId);
  }

  revalidatePath("/journal");
  return { success: true, crisis };
}
