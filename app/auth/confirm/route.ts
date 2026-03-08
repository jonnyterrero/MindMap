import { type EmailOtpType } from "@supabase/supabase-js";
import { type NextRequest, NextResponse } from "next/server";
import { createClient } from "@/lib/supabase-server";

// Supabase redirects here after a user clicks their confirmation email.
// The URL contains a one-time token that must be exchanged for a session.
export async function GET(request: NextRequest) {
  const { searchParams, origin } = new URL(request.url);

  const token_hash = searchParams.get("token_hash");
  const type = searchParams.get("type") as EmailOtpType | null;
  const next = searchParams.get("next") ?? "/today";

  // Reject malformed confirmation links
  if (!token_hash || !type) {
    return NextResponse.redirect(`${origin}/login?error=invalid_confirmation_link`);
  }

  const supabase = await createClient();

  const { error } = await supabase.auth.verifyOtp({
    type,
    token_hash,
  });

  if (error) {
    console.error("[auth/confirm] OTP verification failed:", error.message);
    return NextResponse.redirect(
      `${origin}/login?error=${encodeURIComponent(error.message)}`
    );
  }

  // Successful confirmation — redirect to the app
  return NextResponse.redirect(`${origin}${next}`);
}
