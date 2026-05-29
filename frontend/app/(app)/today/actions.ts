"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";
import { calculateMindMapScore } from "@/lib/mindmap-score";
import { syncTodayWeather } from "@/app/(app)/settings/actions";

const DEFAULT_CHECKIN_CARDS = [
  "sleep",
  "mood",
  "focus",
  "migraine",
  "medication",
  "routines",
  "journal",
];

export type EntryPayload = {
  sleep_minutes: number | null;
  sleep_quality: number | null;
  bed_time: string | null;
  wake_time: string | null;
  mood_valence: number | null;
  anxiety: number | null;
  depression: number | null;
  mania: number | null;
  focus: number | null;
  productivity: number | null;
  migraine: boolean;
  migraine_intensity: number | null;
  migraine_aura: boolean | null;
  notes: string | null;
};

export async function upsertTodayEntry(payload: EntryPayload) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { error: "Not authenticated" };
  }

  const today = new Date().toISOString().split("T")[0];

  const { data: existing } = await supabase
    .from("mindmap_entries")
    .select("id")
    .eq("user_id", user.id)
    .eq("entry_date", today)
    .maybeSingle();

  if (existing) {
    const { error } = await supabase
      .from("mindmap_entries")
      .update({ ...payload, updated_at: new Date().toISOString() })
      .eq("id", existing.id);

    if (error) return { error: error.message };
  } else {
    const { error } = await supabase.from("mindmap_entries").insert({
      user_id: user.id,
      entry_date: today,
      ...payload,
    });

    if (error) return { error: error.message };
  }

  revalidatePath("/today");
  return { success: true };
}

export async function getBodySensations(entryId: string) {
  const supabase = await createClient();
  const { data } = await supabase
    .from("mindmap_body_sensations")
    .select("*")
    .eq("entry_id", entryId)
    .order("created_at", { ascending: true });

  return data ?? [];
}

export async function addBodySensation(
  bodyPart: string,
  sensation: string,
  intensity: number
) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const today = new Date().toISOString().split("T")[0];
  let { data: entry } = await supabase
    .from("mindmap_entries")
    .select("id")
    .eq("user_id", user.id)
    .eq("entry_date", today)
    .maybeSingle();

  if (!entry) {
    const { data: newEntry, error: insertError } = await supabase
      .from("mindmap_entries")
      .insert({ user_id: user.id, entry_date: today })
      .select("id")
      .single();
    if (insertError) return { error: insertError.message };
    entry = newEntry;
  }

  const { error } = await supabase.from("mindmap_body_sensations").insert({
    entry_id: entry!.id,
    body_part: bodyPart,
    sensation,
    intensity,
  });

  if (error) return { error: error.message };
  revalidatePath("/today");
  return { success: true };
}

export async function removeBodySensation(id: string) {
  const supabase = await createClient();
  const { error } = await supabase
    .from("mindmap_body_sensations")
    .delete()
    .eq("id", id);

  if (error) return { error: error.message };
  revalidatePath("/today");
  return { success: true };
}

export async function getTodayEntry() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) return null;

  const today = new Date().toISOString().split("T")[0];

  const { data } = await supabase
    .from("mindmap_entries")
    .select("*")
    .eq("user_id", user.id)
    .eq("entry_date", today)
    .maybeSingle();

  return data;
}

export async function getActiveRoutinesWithStatus() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) return [];

  const today = new Date().toISOString().split("T")[0];

  const { data: routines } = await supabase
    .from("mindmap_routines")
    .select("id, name")
    .eq("user_id", user.id)
    .eq("is_active", true)
    .order("created_at", { ascending: true });

  if (!routines || routines.length === 0) return [];

  const { data: entry } = await supabase
    .from("mindmap_entries")
    .select("id")
    .eq("user_id", user.id)
    .eq("entry_date", today)
    .maybeSingle();

  if (!entry) {
    return routines.map((r) => ({ ...r, completed: false }));
  }

  const { data: completions } = await supabase
    .from("mindmap_entry_routines")
    .select("routine_id, completed")
    .eq("entry_id", entry.id);

  const completionMap = new Map(
    (completions ?? []).map((c) => [c.routine_id, c.completed])
  );

  return routines.map((r) => ({
    ...r,
    completed: completionMap.get(r.id) ?? false,
  }));
}

export async function toggleRoutineCompletion(
  routineId: string,
  completed: boolean
) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) return { error: "Not authenticated" };

  const today = new Date().toISOString().split("T")[0];

  let { data: entry } = await supabase
    .from("mindmap_entries")
    .select("id")
    .eq("user_id", user.id)
    .eq("entry_date", today)
    .maybeSingle();

  if (!entry) {
    const { data: newEntry, error: insertError } = await supabase
      .from("mindmap_entries")
      .insert({ user_id: user.id, entry_date: today })
      .select("id")
      .single();

    if (insertError) return { error: insertError.message };
    entry = newEntry;
  }

  const { error } = await supabase.from("mindmap_entry_routines").upsert(
    {
      entry_id: entry!.id,
      routine_id: routineId,
      completed,
    },
    { onConflict: "entry_id,routine_id" }
  );

  if (error) return { error: error.message };

  revalidatePath("/today");
  return { success: true };
}

/**
 * Which guided check-in sections this user enabled during onboarding.
 * Falls back to all cards if the profile row is missing.
 */
export async function getCheckinConfig(): Promise<{
  cards: string[];
  focus: string | null;
}> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return { cards: DEFAULT_CHECKIN_CARDS, focus: null };

  const { data } = await supabase
    .from("profiles")
    .select("selected_checkin_cards, selected_focus")
    .eq("id", user.id)
    .maybeSingle();

  return {
    cards:
      data?.selected_checkin_cards && data.selected_checkin_cards.length > 0
        ? (data.selected_checkin_cards as string[])
        : DEFAULT_CHECKIN_CARDS,
    focus: (data?.selected_focus as string | null) ?? null,
  };
}

/** Distinct check-in days completed (one entry == one day). Drives plan progress. */
export async function getCheckInCount(): Promise<number> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return 0;

  const { count } = await supabase
    .from("mindmap_entries")
    .select("id", { count: "exact", head: true })
    .eq("user_id", user.id);

  return count ?? 0;
}

/**
 * Save the guided daily check-in, then compute and store the MindMap Score.
 * The score rewards tracking consistency only — it never reflects health.
 * Signals from related tables (routines, meds, body sensations) are read
 * server-side so the score is accurate regardless of save order.
 */
export async function saveCheckIn(payload: EntryPayload) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const today = new Date().toISOString().split("T")[0];

  const { data: existing } = await supabase
    .from("mindmap_entries")
    .select("id")
    .eq("user_id", user.id)
    .eq("entry_date", today)
    .maybeSingle();

  let entryId = existing?.id as string | undefined;
  const created = !existing;

  if (entryId) {
    const { error } = await supabase
      .from("mindmap_entries")
      .update({ ...payload, updated_at: new Date().toISOString() })
      .eq("id", entryId);
    if (error) return { error: error.message };
  } else {
    const { data: inserted, error } = await supabase
      .from("mindmap_entries")
      .insert({ user_id: user.id, entry_date: today, ...payload })
      .select("id")
      .single();
    if (error) return { error: error.message };
    entryId = inserted.id;
  }

  // Gather consistency signals from related tables for this entry.
  const [routinesRes, medsRes, sensationsRes] = await Promise.all([
    supabase
      .from("mindmap_entry_routines")
      .select("id")
      .eq("entry_id", entryId)
      .eq("completed", true)
      .limit(1),
    supabase
      .from("mindmap_medication_adherence")
      .select("id")
      .eq("entry_id", entryId)
      .eq("was_taken", true)
      .limit(1),
    supabase
      .from("mindmap_body_sensations")
      .select("id")
      .eq("entry_id", entryId)
      .limit(1),
  ]);

  const score = calculateMindMapScore(payload, {
    routineLogged: (routinesRes.data?.length ?? 0) > 0,
    medicationLogged: (medsRes.data?.length ?? 0) > 0,
    bodySymptomLogged: (sensationsRes.data?.length ?? 0) > 0,
    journalLogged: Boolean(payload.notes && payload.notes.trim().length > 0),
  });

  await supabase
    .from("mindmap_entries")
    .update({ mindmap_score: score })
    .eq("id", entryId);

  // Best-effort: pull today's weather snapshot if the user opted in.
  try {
    await syncTodayWeather();
  } catch {
    // Weather is strictly optional; never block a check-in on it.
  }

  revalidatePath("/today");
  revalidatePath("/home");
  return { success: true, score, created };
}
