"use client";

import { RouteError } from "@/components/route-error";

// Segment-level boundary: catches errors from any page in the (app) group that
// doesn't define its own error.tsx (today, journal, insights, body-map, etc.).
// More specific route error.tsx files (companion/provider/reports) take
// precedence for their own routes.
export default function AppError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return <RouteError error={error} reset={reset} />;
}
