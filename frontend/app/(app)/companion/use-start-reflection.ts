"use client";

import { useRouter } from "next/navigation";
import { useTransition } from "react";
import { createConversation } from "./actions";

/**
 * Create a new Companion conversation and navigate into it. Shared by the
 * in-page "New reflection" button and the global floating entry point so the
 * two behave identically.
 */
export function useStartReflection() {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();

  function start() {
    startTransition(async () => {
      const r = await createConversation();
      if ("id" in r) router.push(`/companion/${r.id}`);
    });
  }

  return { start, isPending };
}
