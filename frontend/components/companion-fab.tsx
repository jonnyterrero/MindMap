"use client";

import { usePathname } from "next/navigation";
import { MessageCircle, Loader2 } from "lucide-react";
import { useStartReflection } from "@/app/(app)/companion/use-start-reflection";
import { cn } from "@/lib/utils";

/**
 * Global quick-access entry point to the Companion. Starts a fresh reflection
 * from anywhere in the app. Hidden on /companion itself (the page already has
 * its own "New reflection" button) and below the update banner (z-40 < z-50).
 *
 * Mobile: sits above the bottom dock, including the iOS home-indicator inset.
 * Desktop: bottom-right corner.
 */
export function CompanionFab() {
  const pathname = usePathname();
  const { start, isPending } = useStartReflection();

  // Hide on Companion (it has its own new-chat control) and on Check-In,
  // where this button covers the sticky "Complete check-in" action.
  if (
    pathname === "/companion" ||
    pathname.startsWith("/companion/") ||
    pathname === "/today" ||
    pathname.startsWith("/today/")
  ) {
    return null;
  }

  return (
    <button
      type="button"
      onClick={start}
      disabled={isPending}
      aria-label="Talk to your Companion"
      className={cn(
        "fixed right-4 z-40 md:right-6 md:bottom-6",
        "bottom-[calc(6.75rem+env(safe-area-inset-bottom,0px))]",
        "flex h-14 w-14 items-center justify-center rounded-full",
        "bg-primary text-primary-foreground shadow-lg ring-4 ring-background/60",
        "transition-all hover:brightness-110 active:scale-95",
        "disabled:opacity-70",
      )}
    >
      {isPending ? (
        <Loader2 className="h-6 w-6 animate-spin" aria-hidden="true" />
      ) : (
        <MessageCircle className="h-6 w-6" aria-hidden="true" />
      )}
    </button>
  );
}
