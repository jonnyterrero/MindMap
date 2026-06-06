"use client";

import { useOfflineSync } from "@/hooks/use-offline-sync";
import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";

/** Small header dot: green = synced, amber = pending, red = offline w/ pending. */
export function SyncIndicator() {
  const { isOnline, pendingCount, syncing, syncNow } = useOfflineSync();

  // Nothing to show when online and fully synced.
  if (isOnline && pendingCount === 0 && !syncing) return null;

  const color = !isOnline ? "bg-red-500" : pendingCount > 0 ? "bg-amber-500" : "bg-green-500";
  const label = !isOnline
    ? `Offline${pendingCount ? ` · ${pendingCount} queued` : ""}`
    : syncing
      ? "Syncing…"
      : `${pendingCount} to sync`;

  return (
    <button
      type="button"
      onClick={() => isOnline && syncNow()}
      className="flex items-center gap-1.5 rounded-full border px-2 py-1 text-xs text-muted-foreground"
      title={label}
    >
      {syncing ? <Loader2 className="h-3 w-3 animate-spin" /> : <span className={cn("h-2 w-2 rounded-full", color)} />}
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}
