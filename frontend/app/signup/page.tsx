"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { signUp, resendConfirmation } from "@/app/login/actions";
import { Brain, MailCheck } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function SignUpPage() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  // Set once signUp reports confirmation is required — swaps the form for the
  // "check your inbox" screen below.
  const [sentTo, setSentTo] = useState<string | null>(null);
  const [resendMsg, setResendMsg] = useState<string | null>(null);
  const [resending, setResending] = useState(false);

  async function handleSubmit(formData: FormData) {
    setError(null);

    // Client-side password confirmation check before hitting the server
    const password = formData.get("password") as string;
    const confirmPassword = formData.get("confirmPassword") as string;
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    const result = await signUp(formData);
    // With email confirmation on, signUp returns { confirmEmail, email } rather
    // than redirecting. With it off, it redirects and we never reach here.
    if (result?.error) {
      setError(result.error);
      setLoading(false);
    } else if (result?.confirmEmail) {
      setSentTo(result.email ?? (formData.get("email") as string));
      setLoading(false);
    }
  }

  async function handleResend() {
    if (!sentTo) return;
    setResending(true);
    setResendMsg(null);
    const result = await resendConfirmation(sentTo);
    setResendMsg(result?.error ?? result?.success ?? null);
    setResending(false);
  }

  if (sentTo) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4">
        <Card className="w-full max-w-md glass-card">
          <CardHeader className="text-center">
            <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-full bg-primary/20">
              <MailCheck className="h-6 w-6 text-primary" />
            </div>
            <CardTitle className="text-2xl">Confirm your email</CardTitle>
            <CardDescription>
              We sent a confirmation link to{" "}
              <span className="font-medium text-foreground">{sentTo}</span>.
              Click it to activate your account, then sign in.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-3">
            {resendMsg && (
              <div className="rounded-md bg-muted px-4 py-3 text-sm text-muted-foreground">
                {resendMsg}
              </div>
            )}
            <p className="text-xs text-muted-foreground text-center leading-relaxed">
              Didn&apos;t get it? Check your spam folder, or resend the link.
            </p>
          </CardContent>

          <CardFooter className="flex flex-col gap-3">
            <Button
              type="button"
              variant="outline"
              className="w-full"
              onClick={handleResend}
              disabled={resending}
            >
              {resending ? "Resending…" : "Resend confirmation email"}
            </Button>
            <button
              type="button"
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => router.push("/login")}
            >
              Back to <span className="text-primary font-medium">Sign in</span>
            </button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <Card className="w-full max-w-md glass-card">
        <CardHeader className="text-center">
          <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-full bg-primary/20">
            <Brain className="h-6 w-6 text-primary" />
          </div>
          <CardTitle className="text-2xl">Create Account</CardTitle>
          <CardDescription>
            Start tracking your mental wellness journey.
            <span className="mt-2 block text-xs text-muted-foreground">
              MindMap is a self-tracking tool, not medical advice.
            </span>
          </CardDescription>
        </CardHeader>

        <form action={handleSubmit}>
          <CardContent className="space-y-4">
            {error && (
              <div className="rounded-md bg-destructive/10 border border-destructive/20 px-4 py-3 text-sm text-destructive">
                {error}
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                name="email"
                type="email"
                placeholder="you@example.com"
                required
                autoComplete="email"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                name="password"
                type="password"
                placeholder="At least 6 characters"
                required
                minLength={6}
                autoComplete="new-password"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="confirmPassword">Confirm Password</Label>
              <Input
                id="confirmPassword"
                name="confirmPassword"
                type="password"
                placeholder="Repeat your password"
                required
                minLength={6}
                autoComplete="new-password"
              />
            </div>
          </CardContent>

          <CardFooter className="flex flex-col gap-3">
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "Creating account…" : "Create Account"}
            </Button>

            {/*
              Legal notice at the point of consent: the four documents users
              accept implicitly by creating an account are linked here, so the
              agreement isn't a fiction. The explicit per-clause checkboxes are
              on /consent, which the (app) layout redirects to on first sign-in.
            */}
            <p className="text-xs text-muted-foreground text-center leading-relaxed">
              By creating an account, you agree to our{" "}
              <Link href="/terms" className="underline hover:text-foreground">Terms</Link>
              {" and "}
              <Link href="/privacy" className="underline hover:text-foreground">Privacy Policy</Link>
              , and acknowledge the{" "}
              <Link href="/medical-disclaimer" className="underline hover:text-foreground">Medical Disclaimer</Link>
              {" and "}
              <Link href="/ai-disclosure" className="underline hover:text-foreground">AI Disclosure</Link>.
            </p>

            <button
              type="button"
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => router.push("/login")}
            >
              Already have an account?{" "}
              <span className="text-primary font-medium">Sign in</span>
            </button>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
