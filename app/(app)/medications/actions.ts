"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";

export type MedSchedulePayload = {
  name: string;
  dosage: string | null;
  dose_mg: number | null;
  frequency: string;
  reminder_time: string | null;
  start_date: string | null;
  end_date: string | null;
  notes: string | null;
};

export async function getMedicationSchedules() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) return [];

  const { data } = await supabase
    .from("mindmap_medication_schedule")
    .select("*")
    .eq("user_id", user.id)
    .order("is_active", { ascending: false })
    .order("name", { ascending: true });

  return data ?? [];
}

export async function createMedSchedule(payload: MedSchedulePayload) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase.from("mindmap_medication_schedule").insert({
    user_id: user.id,
    ...payload,
  });

  if (error) return { error: error.message };

  revalidatePath("/medications");
  revalidatePath("/today");
  return { success: true };
}

export async function updateMedSchedule(
  id: string,
  payload: Partial<MedSchedulePayload>
) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("mindmap_medication_schedule")
    .update(payload)
    .eq("id", id)
    .eq("user_id", user.id);

  if (error) return { error: error.message };

  revalidatePath("/medications");
  return { success: true };
}

export async function toggleMedActive(id: string, isActive: boolean) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("mindmap_medication_schedule")
    .update({ is_active: isActive })
    .eq("id", id)
    .eq("user_id", user.id);

  if (error) return { error: error.message };

  revalidatePath("/medications");
  revalidatePath("/today");
  return { success: true };
}

export async function deleteMedSchedule(id: string) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("mindmap_medication_schedule")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id);

  if (error) return { error: error.message };

  revalidatePath("/medications");
  revalidatePath("/today");
  return { success: true };
}

export async function getTodayAdherence() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) return [];

  const today = new Date().toISOString().split("T")[0];

  const { data: schedules } = await supabase
    .from("mindmap_medication_schedule")
    .select("id, name, dosage, dose_mg, reminder_time")
    .eq("user_id", user.id)
    .eq("is_active", true);

  if (!schedules || schedules.length === 0) return [];

  const { data: entry } = await supabase
    .from("mindmap_entries")
    .select("id")
    .eq("user_id", user.id)
    .eq("entry_date", today)
    .maybeSingle();

  const { data: adherenceRows } = await supabase
    .from("mindmap_medication_adherence")
    .select("medication_schedule_id, was_taken, was_skipped, taken_at, notes")
    .in(
      "medication_schedule_id",
      schedules.map((s) => s.id)
    )
    .eq("entry_id", entry?.id ?? "00000000-0000-0000-0000-000000000000");

  const adherenceMap = new Map(
    (adherenceRows ?? []).map((a) => [a.medication_schedule_id, a])
  );

  return schedules.map((s) => ({
    ...s,
    was_taken: adherenceMap.get(s.id)?.was_taken ?? false,
    was_skipped: adherenceMap.get(s.id)?.was_skipped ?? false,
    taken_at: adherenceMap.get(s.id)?.taken_at ?? null,
  }));
}

export async function logMedAdherence(
  scheduleId: string,
  wasTaken: boolean,
  wasSkipped: boolean
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

  const { data: schedule } = await supabase
    .from("mindmap_medication_schedule")
    .select("reminder_time")
    .eq("id", scheduleId)
    .single();

  const scheduledTime = schedule?.reminder_time ?? "08:00";

  const { error } = await supabase.from("mindmap_medication_adherence").upsert(
    {
      medication_schedule_id: scheduleId,
      entry_id: entry!.id,
      scheduled_time: scheduledTime,
      was_taken: wasTaken,
      was_skipped: wasSkipped,
      taken_at: wasTaken ? new Date().toISOString() : null,
    },
    { onConflict: "medication_schedule_id,entry_id" }
  );

  if (error) return { error: error.message };

  revalidatePath("/today");
  revalidatePath("/medications");
  return { success: true };
}
