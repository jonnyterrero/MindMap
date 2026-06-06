"use client";

import { useCallback, useEffect, useState } from "react";
import { getQueuedEntries, removeQueued, queueCount } from "@/lib/offline-queue";
import { syncQueuedEntries } from "@/app/(app)/journal/sync-actions";

/**
 * Offline-first sync for journal writes. Tracks connectivity + pending queue,
 * auto-flushes on reconnect. Wrap writes: when offline, enqueue; when online,
 * write directly (callers do the direct write themselves).
 */
export function useOfflineSync() {
  const [isOnline, setIsOnline] = useState(true);
  const [pendingCount, setPendingCount] = useState(0);
  const [syncing, setSyncing] = useState(false);

  const refresh = useCallback(() => setPendingCount(queueCount()), []);

  const syncNow = useCallback(async () => {
    const items = getQueuedEntries();
    if (items.length === 0) return;
    setSyncing(true);
    try {
      const r = await syncQueuedEntries(items.map((i) => i.payload));
      if (!("error" in r)) removeQueued(items.map((i) => i.localId));
    } finally {
      setSyncing(false);
      setPendingCount(queueCount());
    }
  }, []);

  useEffect(() => {
    setIsOnline(navigator.onLine);
    refresh();
    const onOnline = () => {
      setIsOnline(true);
      void syncNow();
    };
    const onOffline = () => setIsOnline(false);
    window.addEventListener("online", onOnline);
    window.addEventListener("offline", onOffline);
    const t = setInterval(refresh, 4000);
    // Opportunistic flush on mount if anything is pending.
    if (navigator.onLine) void syncNow();
    return () => {
      window.removeEventListener("online", onOnline);
      window.removeEventListener("offline", onOffline);
      clearInterval(t);
    };
  }, [refresh, syncNow]);

  return { isOnline, pendingCount, syncing, syncNow };
}
