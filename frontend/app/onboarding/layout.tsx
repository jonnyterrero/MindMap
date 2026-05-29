import type React from "react";
import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase-server";

// /onboarding lives outside the (app) group for the same reason /consent does:
// rendering AppNav here would prefetch primary links, each re-entering
// (app)/layout.tsx and bouncing off the onboarding gate. This layout only
// enforces authentication and renders the flow on its own.
export default async function OnboardingLayout({
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

  return (
    <div className="min-h-screen bg-background safe-area-inset">
      <main className="container mx-auto flex min-h-screen max-w-lg flex-col px-4 py-8">
        {children}
      </main>
    </div>
  );
}
