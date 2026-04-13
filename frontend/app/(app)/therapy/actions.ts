"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";

export type TherapyPayload = {
  session_date: string;
  session_time: string | null;
  duration_minutes: number | null;
  therapist_name: string | null;
  session_type: string;
  notes: string | null;
  mood_before: number | null;
  mood_after: number | null;
  topics_discussed: string[];
  homework_assigned: string | null;
  next_session_date: string | null;
};

export async function getTherapySessions() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];

  const { data } = await supabase
    .from("mindmap_therapy_sessions")
    .select("*")
    .eq("user_id", user.id)
    .order("session_date", { ascending: false })
    .limit(50);

  return data ?? [];
}

export async function createTherapySession(payload: TherapyPayload) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase.from("mindmap_therapy_sessions").insert({
    user_id: user.id,
    ...payload,
  });

  if (error) return { error: error.message };
  revalidatePath("/therapy");
  return { success: true };
}

export async function deleteTherapySession(id: string) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("mindmap_therapy_sessions")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id);

  if (error) return { error: error.message };
  revalidatePath("/therapy");
  return { success: true };
}
