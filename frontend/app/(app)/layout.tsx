import type React from "react";
import { createClient } from "@/lib/supabase-server";
import { redirect } from "next/navigation";
import { AppNav } from "@/components/app-nav";

export default async function AppLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  // /consent lives outside this route group, so there's no risk of looping
  // here. If the user hasn't consented, send them there once.
  const { data: consent } = await supabase
    .from("consent_records")
    .select("id")
    .eq("user_id", user.id)
    .eq("consent_type", "terms_of_service")
    .eq("consent_given", true)
    .limit(1)
    .maybeSingle();

  if (!consent) {
    redirect("/consent");
  }

  // Onboarding gate: new users build their check-in before entering the app.
  // /onboarding lives outside this route group, so there's no redirect loop.
  const { data: profile } = await supabase
    .from("profiles")
    .select("onboarding_complete")
    .eq("id", user.id)
    .maybeSingle();

  if (!profile?.onboarding_complete) {
    redirect("/onboarding");
  }

  return (
    <div className="min-h-screen">
      <AppNav user={user} />
      <main className="container mx-auto max-w-4xl px-4 py-6 safe-area-bottom">
        {children}
      </main>
    </div>
  );
}
