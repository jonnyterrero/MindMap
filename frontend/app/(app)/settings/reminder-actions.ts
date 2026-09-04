"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";
import type { ReminderRow } from "@/lib/notifications";

const REMINDER_TYPES = ["Medication", "Routine", "Therapy", "Journal", "Check-in", "Custom"] as const;
export type ReminderType = (typeof REMINDER_TYPES)[number];

const SELECT_COLUMNS = "id, title, description, reminder_type, reminder_time, days_of_week, is_active";

export async function getReminders(): Promise<ReminderRow[]> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];

  const { data } = await supabase
    .from("mindmap_reminders")
    .select(SELECT_COLUMNS)
    .eq("user_id", user.id)
    .order("reminder_time", { ascending: true });

  return (data as ReminderRow[] | null) ?? [];
}

export async function createReminder(input: {
  title: string;
  reminderType: ReminderType;
  reminderTime: string; // "HH:MM"
  daysOfWeek: number[] | null; // 1=Sun … 7=Sat, null = every day
}): Promise<{ error: string } | { reminders: ReminderRow[] }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const title = input.title.trim();
  if (!title) return { error: "Give the reminder a name." };
  if (!REMINDER_TYPES.includes(input.reminderType)) return { error: "Unknown reminder type." };
  if (!/^\d{2}:\d{2}$/.test(input.reminderTime)) return { error: "Pick a time." };
  if (input.daysOfWeek && input.daysOfWeek.some((d) => d < 1 || d > 7)) {
    return { error: "Invalid day selection." };
  }

  const { error } = await supabase.from("mindmap_reminders").insert({
    user_id: user.id,
    title,
    reminder_type: input.reminderType,
    reminder_time: `${input.reminderTime}:00`,
    days_of_week: input.daysOfWeek?.length ? input.daysOfWeek : null,
    is_active: true,
  });
  if (error) return { error: error.message };

  revalidatePath("/settings");
  return { reminders: await getReminders() };
}

export async function setReminderActive(
  id: string,
  isActive: boolean,
): Promise<{ error: string } | { reminders: ReminderRow[] }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("mindmap_reminders")
    .update({ is_active: isActive })
    .eq("id", id)
    .eq("user_id", user.id);
  if (error) return { error: error.message };

  revalidatePath("/settings");
  return { reminders: await getReminders() };
}

export async function deleteReminder(
  id: string,
): Promise<{ error: string } | { reminders: ReminderRow[] }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("mindmap_reminders")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id);
  if (error) return { error: error.message };

  revalidatePath("/settings");
  return { reminders: await getReminders() };
}
