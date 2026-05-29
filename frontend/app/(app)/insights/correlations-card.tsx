import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { GitCompareArrows } from "lucide-react";
import type { Correlation } from "@/lib/correlation-engine";

const STRENGTH_STYLE: Record<string, string> = {
  strong: "bg-primary/10 text-primary",
  moderate: "bg-amber-500/10 text-amber-600 dark:text-amber-400",
  weak: "bg-muted text-muted-foreground",
};

/**
 * Renders the strongest correlations. Returns null when there are none, so the
 * section stays hidden until the user has enough data — keeps Insights simple.
 */
export function CorrelationsCard({ correlations }: { correlations: Correlation[] }) {
  if (correlations.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <GitCompareArrows className="h-5 w-5 text-primary" /> Possible patterns
        </CardTitle>
        <CardDescription>
          Associations found across your recent check-ins — possible patterns, not causes.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {correlations.map((c) => (
          <div key={`${c.aKey}-${c.bKey}`} className="rounded-lg border border-border p-3">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium">
                {c.aLabel} &amp; {c.bLabel}
              </span>
              <span
                className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-medium ${STRENGTH_STYLE[c.strength]}`}
              >
                {c.strength} · r={c.r}
              </span>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">{c.statement}</p>
          </div>
        ))}
        <MedicalDisclaimer variant="inline" />
      </CardContent>
    </Card>
  );
}
