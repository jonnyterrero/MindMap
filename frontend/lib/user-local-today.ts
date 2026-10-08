import type { SupabaseClient } from "@supabase/supabase-js";
import { calendarDate, DEFAULT_TIMEZONE } from "@/lib/local-date";

/** Profile timezone, falling back to the schema default. */
export async function userTimeZone(supabase: SupabaseClient, userId: string): Promise<string> {
  const { data } = await supabase
    .from("profiles")
    .select("timezone")
    .eq("id", userId)
    .maybeSingle();
  const tz = data?.timezone;
  return typeof tz === "string" && tz.trim() ? tz.trim() : DEFAULT_TIMEZONE;
}

/** Today's YYYY-MM-DD in the user's profile timezone. */
export async function userCalendarDate(
  supabase: SupabaseClient,
  userId: string,
  now = new Date(),
): Promise<string> {
  return calendarDate(await userTimeZone(supabase, userId), now);
}
