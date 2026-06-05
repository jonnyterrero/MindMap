"use client";

import { useState, useTransition } from "react";
import { runPredictionEngine, type PredictionRow } from "./prediction-actions";
import { PredictionCard } from "./prediction-card";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Activity, RefreshCw, Loader2 } from "lucide-react";

export function PredictionsSection({ initial }: { initial: PredictionRow[] }) {
  const [predictions, setPredictions] = useState<PredictionRow[]>(initial);
  const [isPending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);

  function refresh() {
    setError(null);
    startTransition(async () => {
      const r = await runPredictionEngine();
      if ("error" in r) setError(r.error);
      else setPredictions(r.predictions);
    });
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Activity className="h-5 w-5 text-primary" /> Predictive signals
            </CardTitle>
            <CardDescription>
              Possible risk for the day ahead, from your own patterns. Not a diagnosis.
            </CardDescription>
          </div>
          <Button size="sm" variant="outline" onClick={refresh} disabled={isPending}>
            {isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {error && <p className="text-sm text-destructive">{error}</p>}

        {predictions.length === 0 ? (
          <div className="rounded-lg border border-dashed p-6 text-center">
            <p className="text-sm text-muted-foreground">
              No predictions yet. Log a few daily check-ins, then tap Refresh to generate your first signals.
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {predictions.map((p) => (
              <PredictionCard key={p.id} prediction={p} />
            ))}
          </div>
        )}

        <MedicalDisclaimer variant="compact" />
      </CardContent>
    </Card>
  );
}
