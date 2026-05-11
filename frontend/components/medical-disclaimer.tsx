import { Info, ShieldAlert } from "lucide-react";
import { cn } from "@/lib/utils";

const SHORT_TEXT =
  "MindMap is for self-tracking and wellness reflection only. It does not diagnose, treat, cure, or prevent any medical condition.";

const FULL_TEXT =
  "MindMap is for self-tracking, journaling, and wellness reflection only. It is not a medical device and does not diagnose, treat, cure, or prevent any disease or medical condition. If you are experiencing a medical emergency or mental-health crisis, contact emergency services or a local crisis hotline immediately.";

type Variant = "compact" | "inline" | "full";

interface MedicalDisclaimerProps {
  variant?: Variant;
  className?: string;
}

/**
 * Required wherever MindMap surfaces insights, predictions, AI-generated
 * reflections, or anything a reviewer could mistake for clinical advice.
 */
export function MedicalDisclaimer({
  variant = "inline",
  className,
}: MedicalDisclaimerProps) {
  if (variant === "compact") {
    return (
      <p
        className={cn(
          "text-[11px] leading-snug text-muted-foreground",
          className,
        )}
        role="note"
        aria-label="Medical disclaimer"
      >
        Not medical advice. Self-tracking only.
      </p>
    );
  }

  if (variant === "full") {
    return (
      <aside
        role="note"
        aria-label="Medical disclaimer"
        className={cn(
          "rounded-lg border border-amber-300/50 bg-amber-50/60 p-4 text-sm text-amber-900 dark:border-amber-500/30 dark:bg-amber-950/30 dark:text-amber-100",
          className,
        )}
      >
        <div className="flex gap-3">
          <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
          <div className="space-y-2">
            <p className="font-semibold">Medical disclaimer</p>
            <p>{FULL_TEXT}</p>
          </div>
        </div>
      </aside>
    );
  }

  return (
    <p
      role="note"
      aria-label="Medical disclaimer"
      className={cn(
        "flex items-start gap-2 rounded-md bg-muted/60 px-3 py-2 text-xs text-muted-foreground",
        className,
      )}
    >
      <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
      <span>{SHORT_TEXT}</span>
    </p>
  );
}

export const MEDICAL_DISCLAIMER_SHORT = SHORT_TEXT;
export const MEDICAL_DISCLAIMER_FULL = FULL_TEXT;
