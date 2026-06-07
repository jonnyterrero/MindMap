import { Skeleton } from "@/components/ui/skeleton";

// Segment-level loading UI: shown instantly while any (app) route streams its
// server-rendered content. A generic page skeleton (header + a few cards) that
// matches the app's spacing; individual routes can add their own loading.tsx
// for a closer match.
export default function AppLoading() {
  return (
    <div className="space-y-6" aria-busy="true" aria-live="polite">
      <div className="space-y-2">
        <Skeleton className="h-7 w-44" />
        <Skeleton className="h-4 w-64" />
      </div>
      <div className="space-y-3">
        <Skeleton className="h-28 w-full rounded-xl" />
        <Skeleton className="h-28 w-full rounded-xl" />
        <Skeleton className="h-28 w-full rounded-xl" />
      </div>
      <span className="sr-only">Loading…</span>
    </div>
  );
}
