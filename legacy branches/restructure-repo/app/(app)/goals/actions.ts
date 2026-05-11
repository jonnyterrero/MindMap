"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";

export type GoalPayload = {
  title: string;
  description: string | null;
  category: string;
  target_value: number | null;
  unit: string | null;
  target_date: string | null;
};

export async function getGoals() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];

  const { data } = await supabase
    .from("mindmap_goals")
    .select("*")
    .eq("user_id", user.id)
    .order("is_completed", { ascending: true })
    .order("created_at", { ascending: false });

  return data ?? [];
}

export async function createGoal(payload: GoalPayload) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase.from("mindmap_goals").insert({
    user_id: user.id,
    ...payload,
  });

  if (error) return { error: error.message };
  revalidatePath("/goals");
  return { success: true };
}

export async function updateGoalProgress(id: string, currentValue: number) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("mindmap_goals")
    .update({ current_value: currentValue })
    .eq("id", id)
    .eq("user_id", user.id);

  if (error) return { error: error.message };
  revalidatePath("/goals");
  return { success: true };
}

export async function toggleGoalComplete(id: string, completed: boolean) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("mindmap_goals")
    .update({
      is_completed: completed,
      completed_at: completed ? new Date().toISOString() : null,
    })
    .eq("id", id)
    .eq("user_id", user.id);

  if (error) return { error: error.message };
  revalidatePath("/goals");
  return { success: true };
}

export async function deleteGoal(id: string) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("mindmap_goals")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id);

  if (error) return { error: error.message };
  revalidatePath("/goals");
  return { success: true };
}
