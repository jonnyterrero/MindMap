import type React from "react";
import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase-server";

// /next-steps lives outside the (app) group for the same reason /onboarding
// and /consent do: rendering AppNav here would prefetch primary links, each
// re-entering (app)/layout.tsx. This layout only enforces authentication and
// renders the hand-off screen on its own.
export default async function NextStepsLayout({
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
      <main className="container mx-auto flex min-h-screen max-w-lg flex-col justify-center px-4 py-8">
        {children}
      </main>
    </div>
  );
}
