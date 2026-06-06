"use client";

import { useRouter } from "next/navigation";
import { useTransition } from "react";
import { createConversation } from "./actions";
import { Button } from "@/components/ui/button";
import { Plus, Loader2 } from "lucide-react";

export function NewChatButton() {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();

  function start() {
    startTransition(async () => {
      const r = await createConversation();
      if ("id" in r) router.push(`/companion/${r.id}`);
    });
  }

  return (
    <Button onClick={start} disabled={isPending}>
      {isPending ? <Loader2 className="animate-spin" /> : <Plus />}
      New reflection
    </Button>
  );
}
