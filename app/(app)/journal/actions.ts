"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";

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

export async function createJournalEntry(payload: JournalPayload) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase.from("mindmap_journal_entries").insert({
    user_id: user.id,
    ...payload,
  });

  if (error) return { error: error.message };
  revalidatePath("/journal");
  return { success: true };
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
