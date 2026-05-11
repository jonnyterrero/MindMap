"use server";

import { createClient } from "@/lib/supabase-server";
import { redirect } from "next/navigation";

export async function checkConsentStatus() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return false;

  const { data } = await supabase
    .from("consent_records")
    .select("id")
    .eq("user_id", user.id)
    .eq("consent_type", "terms_of_service")
    .eq("consent_given", true)
    .limit(1)
    .maybeSingle();

  return !!data;
}

export async function grantConsent(consentTypes: string[]) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const records = consentTypes.map((type) => ({
    user_id: user.id,
    consent_type: type,
    consent_given: true,
    consent_version: "1.0",
    ip_address: null,
  }));

  const { error } = await supabase.from("consent_records").insert(records);
  if (error) return { error: error.message };

  redirect("/today");
}
