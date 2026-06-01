"use server";

import { createClient } from "@/lib/supabase-server";
import { revalidatePath } from "next/cache";
import { geocodeCity, fetchDailyWeather } from "@/lib/weather";

export async function getProfile() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return null;

  const { data } = await supabase
    .from("profiles")
    .select("*")
    .eq("id", user.id)
    .maybeSingle();

  return { ...data, email: user.email };
}

export async function updateProfile(displayName: string, timezone: string) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("profiles")
    .update({ display_name: displayName.trim(), timezone })
    .eq("id", user.id);

  if (error) return { error: error.message };

  revalidatePath("/settings");
  return { success: true };
}

export async function changePassword(
  currentPassword: string,
  newPassword: string,
) {
  if (!newPassword || newPassword.length < 8) {
    return { error: "Password must be at least 8 characters." };
  }

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user?.email) return { error: "Not authenticated" };

  // Re-verify the current password before allowing the change.
  // Prevents a hijacked session from silently rotating the password.
  const { error: signInError } = await supabase.auth.signInWithPassword({
    email: user.email,
    password: currentPassword,
  });
  if (signInError) {
    return { error: "Current password is incorrect." };
  }

  const { error } = await supabase.auth.updateUser({ password: newPassword });
  if (error) return { error: error.message };

  return { success: true };
}

/**
 * Enable/disable weather tracking and (when enabling) resolve a city name to
 * coordinates via Open-Meteo. Location is stored on the profile.
 */
export async function updateWeatherSettings(enabled: boolean, city: string) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const patch: Record<string, unknown> = { weather_enabled: enabled };

  if (enabled) {
    const trimmed = city.trim();
    if (trimmed) {
      const geo = await geocodeCity(trimmed);
      if (!geo) {
        return { error: "Couldn't find that location. Try a city like \"Austin, TX\"." };
      }
      patch.weather_lat = geo.lat;
      patch.weather_lon = geo.lon;
      patch.weather_label = geo.label;
    }
  }

  const { error } = await supabase
    .from("profiles")
    .upsert({ id: user.id, ...patch }, { onConflict: "id" });
  if (error) return { error: error.message };

  // Pull today's snapshot right away so the user sees it working.
  if (enabled) await syncTodayWeather();

  revalidatePath("/settings");
  revalidatePath("/insights");
  return { success: true, label: patch.weather_label as string | undefined };
}

/**
 * Best-effort: fetch and store today's weather snapshot if the user opted in
 * and we don't already have it. Safe to call on every page load.
 */
export async function syncTodayWeather() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return;

  const { data: profile } = await supabase
    .from("profiles")
    .select("weather_enabled, weather_lat, weather_lon")
    .eq("id", user.id)
    .maybeSingle();

  if (!profile?.weather_enabled || profile.weather_lat == null || profile.weather_lon == null) {
    return;
  }

  const today = new Date().toISOString().split("T")[0];

  const { data: existing } = await supabase
    .from("mindmap_weather_daily")
    .select("id")
    .eq("user_id", user.id)
    .eq("entry_date", today)
    .maybeSingle();
  if (existing) return;

  const w = await fetchDailyWeather(
    profile.weather_lat as number,
    profile.weather_lon as number,
    today,
  );
  if (!w) return;

  // Compute pressure change vs the most recent prior snapshot, if any.
  let pressureChange: number | null = null;
  if (w.pressure != null) {
    const { data: prev } = await supabase
      .from("mindmap_weather_daily")
      .select("pressure")
      .eq("user_id", user.id)
      .lt("entry_date", today)
      .order("entry_date", { ascending: false })
      .limit(1)
      .maybeSingle();
    if (prev?.pressure != null) pressureChange = w.pressure - (prev.pressure as number);
  }

  await supabase.from("mindmap_weather_daily").upsert(
    {
      user_id: user.id,
      entry_date: today,
      temp_max: w.temp_max,
      temp_min: w.temp_min,
      humidity: w.humidity,
      pressure: w.pressure,
      pressure_change: pressureChange,
      precipitation: w.precipitation,
      weather_code: w.weather_code,
    },
    { onConflict: "user_id,entry_date" },
  );
}

/** Enable/disable opt-in AI journal reflection. */
export async function updateAiReflectionSetting(enabled: boolean) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase
    .from("profiles")
    .upsert({ id: user.id, ai_reflection_enabled: enabled }, { onConflict: "id" });
  if (error) return { error: error.message };

  revalidatePath("/settings");
  revalidatePath("/journal");
  return { success: true };
}

export async function requestDataDeletion(scope: string, reason: string | null) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { error } = await supabase.from("data_deletion_requests").insert({
    user_id: user.id,
    scope,
    reason: reason?.trim() || null,
  });

  if (error) return { error: error.message };

  revalidatePath("/settings");
  return { success: true };
}

export async function getDeletionRequests() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];

  const { data } = await supabase
    .from("data_deletion_requests")
    .select("*")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });

  return data ?? [];
}
