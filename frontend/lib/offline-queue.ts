/**
 * Offline queue (web)
 * -------------------
 * Browser-only localStorage queue for writes made while offline. On the web
 * this is the persistence layer; in the native app the same queue is mirrored
 * to SQLite (@capacitor-community/sqlite) — see MOBILE.md. The server table
 * `mindmap_offline_queue` backs the native sync + conflict tracking.
 */
import type { JournalPayload } from "@/app/(app)/journal/actions";

const KEY = "mindmap_offline_queue_v1";

export type QueuedJournalEntry = {
  localId: string;
  payload: JournalPayload;
  createdAt: string;
};

function read(): QueuedJournalEntry[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(window.localStorage.getItem(KEY) ?? "[]");
  } catch {
    return [];
  }
}

function write(items: QueuedJournalEntry[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(KEY, JSON.stringify(items));
}

export function enqueueJournalEntry(payload: JournalPayload): QueuedJournalEntry {
  const item: QueuedJournalEntry = {
    localId: `local-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    payload,
    createdAt: new Date().toISOString(),
  };
  write([...read(), item]);
  return item;
}

export function getQueuedEntries(): QueuedJournalEntry[] {
  return read();
}

export function removeQueued(localIds: string[]) {
  const set = new Set(localIds);
  write(read().filter((i) => !set.has(i.localId)));
}

export function queueCount(): number {
  return read().length;
}
