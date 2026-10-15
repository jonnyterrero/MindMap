"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { signIn, resendConfirmation } from "./actions";
import { Brain } from "lucide-react";
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

export default function LoginPage() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  // When sign-in fails because the email isn't confirmed, hold the address so
  // we can offer a resend button instead of a dead-end error.
  const [unconfirmedEmail, setUnconfirmedEmail] = useState<string | null>(null);
  const [resendMsg, setResendMsg] = useState<string | null>(null);
  const [resending, setResending] = useState(false);

  async function handleSubmit(formData: FormData) {
    setError(null);
    setResendMsg(null);
    setUnconfirmedEmail(null);
    setLoading(true);
    const result = await signIn(formData);
    // signIn redirects on success — only arrives here on error
    if (result?.error) {
      setError(result.error);
      if (result.needsConfirmation) {
        setUnconfirmedEmail(result.email ?? (formData.get("email") as string));
      }
      setLoading(false);
    }
  }

  async function handleResend() {
    if (!unconfirmedEmail) return;
    setResending(true);
    setResendMsg(null);
    const result = await resendConfirmation(unconfirmedEmail);
    setResendMsg(result?.error ?? result?.success ?? null);
    setResending(false);
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <Card className="w-full max-w-md glass-card">
        <CardHeader className="text-center">
          <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-full bg-primary/20">
            <Brain className="h-6 w-6 text-primary" />
          </div>
          <CardTitle className="text-2xl">Welcome Back</CardTitle>
          <CardDescription>Sign in to your MindMap account</CardDescription>
        </CardHeader>

        <form action={handleSubmit}>
          <CardContent className="space-y-4">
            {error && (
              <div className="rounded-md bg-destructive/10 border border-destructive/20 px-4 py-3 text-sm text-destructive">
                {error}
                {unconfirmedEmail && (
                  <button
                    type="button"
                    onClick={handleResend}
                    disabled={resending}
                    className="mt-2 block font-medium underline hover:no-underline disabled:opacity-60"
                  >
                    {resending ? "Resending…" : "Resend confirmation email"}
                  </button>
                )}
              </div>
            )}

            {resendMsg && (
              <div className="rounded-md bg-muted px-4 py-3 text-sm text-muted-foreground">
                {resendMsg}
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
              <div className="flex items-center justify-between">
                <Label htmlFor="password">Password</Label>
                <Link
                  href="/forgot-password"
                  className="text-xs text-muted-foreground hover:text-primary transition-colors"
                >
                  Forgot password?
                </Link>
              </div>
              <Input
                id="password"
                name="password"
                type="password"
                placeholder="••••••••"
                required
                minLength={6}
                autoComplete="current-password"
              />
            </div>
          </CardContent>

          <CardFooter className="flex flex-col gap-3">
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "Signing in…" : "Sign In"}
            </Button>

            <button
              type="button"
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => router.push("/signup")}
            >
              Don&apos;t have an account?{" "}
              <span className="text-primary font-medium">Sign up</span>
            </button>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
