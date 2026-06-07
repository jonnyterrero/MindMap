"use client";

import { useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { AlertTriangle, ArrowLeft, RotateCw } from "lucide-react";

/**
 * Shared error-boundary UI for route `error.tsx` files. Keeps the recovery
 * affordance (Retry / Home) consistent across the app. Logs to the console so
 * the digest is discoverable in dev and in the browser during triage.
 */
export function RouteError({
  error,
  reset,
  title = "Something went wrong",
  description = "An unexpected error occurred. You can retry or head back home.",
  homeHref = "/home",
}: {
  error: Error & { digest?: string };
  reset: () => void;
  title?: string;
  description?: string;
  homeHref?: string;
}) {
  useEffect(() => {
    console.error("Route error boundary caught:", error);
  }, [error]);

  return (
    <div className="mx-auto max-w-lg py-8">
      <Card>
        <CardHeader className="text-center">
          <div className="mb-2 flex justify-center">
            <div className="rounded-full bg-destructive/10 p-3">
              <AlertTriangle className="h-7 w-7 text-destructive" />
            </div>
          </div>
          <CardTitle>{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-3">
          <div className="flex justify-center gap-2">
            <Button variant="outline" onClick={() => reset()}>
              <RotateCw /> Retry
            </Button>
            <Button asChild>
              <Link href={homeHref}>
                <ArrowLeft /> Home
              </Link>
            </Button>
          </div>
          {error.digest && (
            <p className="text-xs text-muted-foreground">
              Reference: {error.digest}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
