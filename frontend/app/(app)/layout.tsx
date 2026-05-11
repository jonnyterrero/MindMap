import type React from "react";
import { createClient } from "@/lib/supabase-server";
import { redirect } from "next/navigation";
import { headers } from "next/headers";
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

  const headersList = await headers();
  const pathname = headersList.get("x-pathname") ?? "";
  const isConsentPage = pathname === "/consent";

  if (!isConsentPage) {
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
