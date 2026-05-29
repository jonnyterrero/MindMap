"use server";

import { createClient } from "@/lib/supabase-server";
import { redirect } from "next/navigation";
import { FOCUS_OPTIONS, CHECKIN_CARDS, type FocusOption } from "./constants";

export async function checkOnboardingStatus(): Promise<boolean> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return false;

  const { data } = await supabase
    .from("profiles")
    .select("onboarding_complete")
    .eq("id", user.id)
    .maybeSingle();

  return Boolean(data?.onboarding_complete);
}

export async function completeOnboarding(input: {
  focus: string;
  cards: string[];
}) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  if (!FOCUS_OPTIONS.includes(input.focus as FocusOption)) {
    return { error: "Please choose a main focus to continue." };
  }

  // Keep only known cards; always preserve order from CHECKIN_CARDS.
  const cards = CHECKIN_CARDS.filter((c) => input.cards.includes(c));
  if (cards.length === 0) {
    return { error: "Select at least one check-in card." };
  }

  const { error } = await supabase.from("profiles").upsert(
    {
      id: user.id,
      onboarding_complete: true,
      onboarding_completed_at: new Date().toISOString(),
      selected_focus: input.focus,
      selected_checkin_cards: cards,
    },
    { onConflict: "id" },
  );

  if (error) return { error: error.message };

  // On success this throws NEXT_REDIRECT and never returns.
  redirect("/today");
}
