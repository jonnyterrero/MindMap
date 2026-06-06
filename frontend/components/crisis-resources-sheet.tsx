"use client";

import { useTransition } from "react";
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerDescription,
  DrawerFooter,
} from "@/components/ui/drawer";
import { Button } from "@/components/ui/button";
import { acknowledgeCrisisEvent } from "@/app/(app)/crisis-actions";
import { CRISIS_RESOURCES, crisisHeader, type CrisisSeverity } from "@/lib/crisis-detection";
import { Phone, MessageSquare, ShieldAlert, Loader2 } from "lucide-react";

/**
 * Severity-appropriate crisis resources. For `critical`, the sheet is
 * non-dismissible — only the "I'm safe" acknowledgment closes it.
 */
export function CrisisResourcesSheet({
  severity,
  eventId,
  onClose,
}: {
  severity: CrisisSeverity | null;
  eventId?: string | null;
  onClose: () => void;
}) {
  const [isPending, startTransition] = useTransition();
  const open = severity !== null;
  const header = severity ? crisisHeader(severity) : null;
  const dismissible = severity !== "critical";

  function acknowledge() {
    startTransition(async () => {
      if (eventId) await acknowledgeCrisisEvent(eventId);
      onClose();
    });
  }

  return (
    <Drawer open={open} dismissible={dismissible} onOpenChange={(o) => { if (!o && dismissible) onClose(); }}>
      <DrawerContent>
        <div className="mx-auto w-full max-w-md">
          <DrawerHeader>
            <div className="mb-1 flex justify-center">
              <div className="rounded-full bg-red-500/10 p-2.5">
                <ShieldAlert className="h-6 w-6 text-red-600 dark:text-red-400" />
              </div>
            </div>
            <DrawerTitle className="text-center">{header?.title}</DrawerTitle>
            <DrawerDescription className="text-center">{header?.body}</DrawerDescription>
          </DrawerHeader>

          <div className="space-y-2 px-4">
            {CRISIS_RESOURCES.map((r) => (
              <a
                key={r.label}
                href={r.href}
                className="flex items-center gap-3 rounded-lg border border-border p-3 transition-colors hover:bg-muted/60"
              >
                {r.label.includes("Text") ? (
                  <MessageSquare className="h-5 w-5 shrink-0 text-primary" />
                ) : (
                  <Phone className="h-5 w-5 shrink-0 text-primary" />
                )}
                <div>
                  <p className="text-sm font-medium">{r.label}</p>
                  <p className="text-xs text-muted-foreground">{r.detail}</p>
                </div>
              </a>
            ))}
          </div>

          <DrawerFooter>
            <Button onClick={acknowledge} disabled={isPending} variant={severity === "critical" ? "default" : "outline"}>
              {isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              I&apos;m safe right now
            </Button>
          </DrawerFooter>
        </div>
      </DrawerContent>
    </Drawer>
  );
}
