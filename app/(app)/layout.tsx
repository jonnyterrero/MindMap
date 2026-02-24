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

  return (
    <div className="min-h-screen">
      <AppNav user={user} />
      <main className="container mx-auto max-w-4xl px-4 py-6">{children}</main>
    </div>
  );
}
