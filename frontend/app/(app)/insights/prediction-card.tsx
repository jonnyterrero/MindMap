"use client";

import { useState, useTransition } from "react";
import { recordPredictionOutcome, type PredictionRow } from "./prediction-actions";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ChevronDown, ChevronUp, Check, X, Loader2 } from "lucide-react";

const LABELS: Record<string, string> = {
  migraine: "Migraine",
  anxiety: "Anxiety",
  mood: "Mood dip",
  pain_flare: "Pain flare",
};

const LEVEL_STYLE: Record<string, { pill: string; label: string }> = {
  low: { pill: "bg-green-500/15 text-green-600 dark:text-green-400", label: "Low" },
  moderate: { pill: "bg-amber-500/15 text-amber-600 dark:text-amber-400", label: "Moderate" },
  high: { pill: "bg-orange-500/15 text-orange-600 dark:text-orange-400", label: "High" },
  critical: { pill: "bg-red-500/15 text-red-600 dark:text-red-400", label: "Elevated" },
};

export function PredictionCard({ prediction }: { prediction: PredictionRow }) {
  const [open, setOpen] = useState(false);
  const [outcome, setOutcome] = useState<string | null>(prediction.outcome_recorded);
  const [isPending, startTransition] = useTransition();

  const style = LEVEL_STYLE[prediction.risk_level] ?? LEVEL_STYLE.low;
  const factors = (prediction.contributing_factors ?? []).filter((f) => f.detail);

  function mark(o: "accurate" | "inaccurate") {
    setOutcome(o);
    startTransition(async () => {
      const r = await recordPredictionOutcome(prediction.id, o);
      if (r?.error) setOutcome(prediction.outcome_recorded); // revert
    });
  }

  return (
    <Card>
      <CardContent className="p-3">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex w-full items-center justify-between gap-2 text-left"
          aria-expanded={open}
        >
          <div className="flex items-center gap-2">
            <span className={cn("rounded-full px-2 py-0.5 text-xs font-semibold", style.pill)}>
              {style.label}
            </span>
            <span className="text-sm font-medium">{LABELS[prediction.prediction_type] ?? prediction.prediction_type}</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>{Math.round(prediction.confidence * 100)}% conf.</span>
            {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </div>
        </button>

        {open && (
          <div className="mt-3 space-y-3">
            {factors.length > 0 ? (
              <ul className="space-y-1">
                {factors.map((f, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                    <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-muted-foreground" />
                    {f.detail}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-muted-foreground">No notable factors today.</p>
            )}

            <div className="flex items-center gap-2 border-t pt-2">
              {outcome ? (
                <p className="text-xs text-muted-foreground">
                  You marked this <span className="font-medium">{outcome}</span>. Thanks — it helps tune your model.
                </p>
              ) : (
                <>
                  <span className="text-xs text-muted-foreground">Was this right?</span>
                  <Button size="sm" variant="outline" className="h-7 px-2" disabled={isPending} onClick={() => mark("accurate")}>
                    {isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />} Accurate
                  </Button>
                  <Button size="sm" variant="outline" className="h-7 px-2" disabled={isPending} onClick={() => mark("inaccurate")}>
                    <X className="h-3.5 w-3.5" /> No
                  </Button>
                </>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
