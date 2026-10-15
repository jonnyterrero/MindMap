"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { headers } from "next/headers";
import { createClient } from "@/lib/supabase-server";
import { AnalyticsEvent } from "@/lib/analytics-events";
import { captureServerEvent } from "@/lib/analytics-server";

// Resolve the public app origin for email redirect links.
// Prefers NEXT_PUBLIC_APP_URL, falls back to the request origin so we never
// send empty/relative redirect URLs to Supabase Auth.
async function getAppOrigin(): Promise<string> {
  if (process.env.NEXT_PUBLIC_APP_URL) return process.env.NEXT_PUBLIC_APP_URL;
  const h = await headers();
  const host = h.get("x-forwarded-host") ?? h.get("host");
  const proto = h.get("x-forwarded-proto") ?? "https";
  return host ? `${proto}://${host}` : "";
}

// ─── Sign In ────────────────────────────────────────────────────────────────
export async function signIn(formData: FormData) {
  const supabase = await createClient();

  const email = formData.get("email") as string;
  const password = formData.get("password") as string;

  const { error } = await supabase.auth.signInWithPassword({ email, password });

  if (error) {
    // A user whose email isn't confirmed yet can't sign in. Supabase surfaces
    // this as the `email_not_confirmed` code (older versions fold it into the
    // generic "Invalid login credentials"). Give these users a clear message
    // and a flag so the login page can offer to resend the confirmation link,
    // instead of leaving them staring at "invalid credentials" after signup.
    if (error.code === "email_not_confirmed") {
      return {
        error:
          "Your email isn't confirmed yet. Check your inbox for the confirmation link — we can resend it below.",
        needsConfirmation: true,
        email,
      };
    }
    // Return error message to show in the form
    return { error: error.message };
  }

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (user) await captureServerEvent(user.id, AnalyticsEvent.SignedIn);

  revalidatePath("/", "layout");
  redirect("/home");
}

// ─── Sign Up ─────────────────────────────────────────────────────────────────
export async function signUp(formData: FormData) {
  const supabase = await createClient();

  const email = formData.get("email") as string;
  const password = formData.get("password") as string;

  const origin = await getAppOrigin();
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      emailRedirectTo: `${origin}/auth/confirm`,
    },
  });

  if (error) {
    return { error: error.message };
  }

  // When "Confirm email" is enabled in Supabase, signUp creates the user but
  // returns NO session — the account isn't usable until the email link is
  // clicked. Redirecting to /home here (as this did before) dropped the user
  // into a route with no auth cookie, so the (app) layout immediately bounced
  // them back to /login and left the client router cache stale (the "had to
  // refresh for the page to load" symptom). Instead, tell the UI to show a
  // "check your inbox" state and stop.
  //
  // This branch also covers Supabase's email-enumeration guard: signing up an
  // address that already exists returns a user with no session and no error.
  // Showing the same confirmation screen avoids leaking which emails are
  // registered.
  if (!data.session) {
    if (data.user) await captureServerEvent(data.user.id, AnalyticsEvent.SignedUp);
    return { confirmEmail: true, email };
  }

  // Confirmation disabled — a real session exists, so the user is signed in.
  // Home is the daily hub; the app layout still sends new accounts through
  // consent and onboarding before they can stay there.
  if (data.user) await captureServerEvent(data.user.id, AnalyticsEvent.SignedUp);
  revalidatePath("/", "layout");
  redirect("/home");
}

// ─── Resend Confirmation Email ───────────────────────────────────────────────
export async function resendConfirmation(email: string) {
  if (!email) return { error: "Email is required." };

  const supabase = await createClient();
  const origin = await getAppOrigin();

  const { error } = await supabase.auth.resend({
    type: "signup",
    email,
    options: { emailRedirectTo: `${origin}/auth/confirm` },
  });

  if (error) return { error: error.message };

  // Same neutral message whether or not the address has a pending signup, so
  // this doesn't leak which emails are registered.
  return {
    success:
      "If that account needs confirming, we've sent a new link. Check your inbox (and spam).",
  };
}

// ─── Sign Out ─────────────────────────────────────────────────────────────────
export async function signOut() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (user) await captureServerEvent(user.id, AnalyticsEvent.SignedOut);
  await supabase.auth.signOut();
  revalidatePath("/", "layout");
  redirect("/login");
}

// ─── Forgot Password ─────────────────────────────────────────────────────────
export async function forgotPassword(formData: FormData) {
  const supabase = await createClient();
  const email = formData.get("email") as string;

  if (!email) return { error: "Email is required." };

  const origin = await getAppOrigin();
  const { error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: `${origin}/auth/confirm?next=/reset-password`,
  });

  if (error) return { error: error.message };

  // Always return the same success message so the endpoint doesn't leak
  // which addresses are registered.
  return {
    success:
      "If an account exists for that email, we've sent password reset instructions.",
  };
}

// ─── Update Password (after recovery OTP) ────────────────────────────────────
export async function updatePassword(formData: FormData) {
  const supabase = await createClient();
  const password = formData.get("password") as string;
  const confirm = formData.get("confirm") as string;

  if (!password || password.length < 8) {
    return { error: "Password must be at least 8 characters." };
  }
  if (password !== confirm) {
    return { error: "Passwords do not match." };
  }

  const { error } = await supabase.auth.updateUser({ password });
  if (error) return { error: error.message };

  revalidatePath("/", "layout");
  redirect("/home");
}
