import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

// Shared building blocks for per-route loading.tsx files. Every (app) page
// follows the same shell — a title/subtitle header then content — so the
// skeletons compose from these rather than each route hand-rolling its own
// spacing and accessibility attributes.

/**
 * Outer wrapper for a route skeleton. Applies the app's page spacing plus the
 * busy/live semantics, and hides the decorative blocks from screen readers in
 * favour of a single announced "Loading" message.
 */
export function PageSkeleton({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("space-y-6", className)} aria-busy="true" aria-live="polite">
      <div aria-hidden="true" className={cn("space-y-6", className)}>
        {children}
      </div>
      <span className="sr-only">Loading…</span>
    </div>
  );
}

/** Page title + one line of description, matching the h1/muted-p header. */
export function PageHeaderSkeleton({ titleClassName }: { titleClassName?: string }) {
  return (
    <div className="space-y-2">
      <Skeleton className={cn("h-7 w-44", titleClassName)} />
      <Skeleton className="h-4 w-64" />
    </div>
  );
}

/** A vertical stack of equally sized cards. */
export function CardStackSkeleton({
  count = 3,
  className,
}: {
  count?: number;
  className?: string;
}) {
  return (
    <div className="space-y-3">
      {Array.from({ length: count }, (_, i) => (
        <Skeleton key={i} className={cn("h-28 w-full rounded-xl", className)} />
      ))}
    </div>
  );
}

/** Compact rows, for list-style pages (medications, routines, goals, reports). */
export function ListRowsSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: count }, (_, i) => (
        <Skeleton key={i} className="h-16 w-full rounded-lg" />
      ))}
    </div>
  );
}

/** Two-column stat tiles, as used by the baseline report. */
export function StatGridSkeleton({ count = 8 }: { count?: number }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {Array.from({ length: count }, (_, i) => (
        <Skeleton key={i} className="h-20 w-full rounded-lg" />
      ))}
    </div>
  );
}

/** A tall block standing in for a chart, canvas or diagram. */
export function CanvasSkeleton({ className }: { className?: string }) {
  return <Skeleton className={cn("h-72 w-full rounded-xl", className)} />;
}
