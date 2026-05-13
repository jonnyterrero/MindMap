import type React from "react";
import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase-server";

// /consent lives outside the (app) group on purpose: rendering AppNav here
// would cause Next to prefetch every primary link, each prefetch would enter
// (app)/layout.tsx, fail the consent gate, and redirect back to /consent —
// flooding the response stream and producing a blank post-login screen.
//
// This layout only enforces "must be signed in" and renders the form alone.
export default async function ConsentLayout({
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
      <main className="container mx-auto max-w-2xl px-4 py-10">
        {children}
      </main>
    </div>
  );
}
