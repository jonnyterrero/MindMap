"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase-server";

// ─── Sign In ────────────────────────────────────────────────────────────────
export async function signIn(formData: FormData) {
  const supabase = await createClient();

  const email = formData.get("email") as string;
  const password = formData.get("password") as string;

  const { error } = await supabase.auth.signInWithPassword({ email, password });

  if (error) {
    // Return error message to show in the form
    return { error: error.message };
  }

  revalidatePath("/", "layout");
  redirect("/today");
}

// ─── Sign Up ─────────────────────────────────────────────────────────────────
export async function signUp(formData: FormData) {
  const supabase = await createClient();

  const email = formData.get("email") as string;
  const password = formData.get("password") as string;

  const { error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      // Where Supabase redirects after email confirmation
      emailRedirectTo: `${process.env.NEXT_PUBLIC_APP_URL}/auth/confirm`,
    },
  });

  if (error) {
    return { error: error.message };
  }

  // If email confirmation is disabled in Supabase, this redirects immediately.
  // If enabled, user gets a confirmation email and lands here after clicking it.
  revalidatePath("/", "layout");
  redirect("/today");
}

// ─── Sign Out ─────────────────────────────────────────────────────────────────
export async function signOut() {
  const supabase = await createClient();
  await supabase.auth.signOut();
  revalidatePath("/", "layout");
  redirect("/login");
}
