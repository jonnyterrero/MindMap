"use client";

import { useStartReflection } from "./use-start-reflection";
import { Button } from "@/components/ui/button";
import { Plus, Loader2 } from "lucide-react";

export function NewChatButton() {
  const { start, isPending } = useStartReflection();

  return (
    <Button onClick={start} disabled={isPending}>
      {isPending ? <Loader2 className="animate-spin" /> : <Plus />}
      New reflection
    </Button>
  );
}
