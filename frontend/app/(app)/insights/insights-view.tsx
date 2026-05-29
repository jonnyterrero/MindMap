"use client";

import { useState, useTransition } from "react";
import { generateInsights } from "./actions";
import { Button } from "@/components/ui/button";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import { RefreshCw, Loader2, AlertTriangle, TrendingUp, Brain, CheckCircle2, Info } from "lucide-react";
import { formatDistanceToNow, parseISO } from "date-fns";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";

type Insight = Record<string, unknown>;

const RISK_CONFIG: Record<string, { color: string; icon: typeof CheckCircle2 }> = {
  low: { color: "text-green-600", icon: CheckCircle2 },
  stable: { color: "text-green-600", icon: CheckCircle2 },
  moderate: { color: "text-yellow-600", icon: Info },
  concerning: { color: "text-orange-600", icon: AlertTriangle },
  high: { color: "text-red-600", icon: AlertTriangle },
  unknown: { color: "text-muted-foreground", icon: Info },
};

export function InsightsView({ insights: initialInsights }: { insights: Insight[] }) {
  const [isPending, startTransition] = useTransition();
  const [insights, setInsights] = useState(initialInsights);

  function handleRefresh() {
    startTransition(async () => {
      const result = await generateInsights();
      if (result && Array.isArray(result)) {
        setInsights(result);
      }
    });
  }

  return (
    <div className="space-y-4">
      <Button variant="outline" onClick={handleRefresh} disabled={isPending}>
        {isPending ? <Loader2 className="animate-spin" /> : <RefreshCw className="h-4 w-4" />}
        Refresh Insights
      </Button>

      <MedicalDisclaimer variant="inline" />

      {insights.length === 0 ? (
        <Card className="glass-card">
          <CardContent className="py-8 text-center text-muted-foreground">
            <Brain className="h-8 w-8 mx-auto mb-3 opacity-40" />
            <p>No insights generated yet.</p>
            <p className="text-sm mt-1">Log a few days of data, then hit Refresh Insights.</p>
          </CardContent>
        </Card>
      ) : (
        insights.map((insight) => {
          const riskLevel = (insight.risk_level as string) ?? "unknown";
          const config = RISK_CONFIG[riskLevel] ?? RISK_CONFIG.unknown;
          const Icon = config.icon;
          const reasons = (insight.reasons as string[]) ?? [];
          const signals = insight.signals as Record<string, unknown> | null;

          return (
            <Card key={insight.id as string} className="glass-card">
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-base flex items-center gap-2">
                      {(insight.insight_type as string) === "migraine_risk" ? (
                        <AlertTriangle className="h-4 w-4 text-primary" />
                      ) : (
                        <TrendingUp className="h-4 w-4 text-primary" />
                      )}
                      {formatInsightType(insight.insight_type as string)}
                    </CardTitle>
                    <CardDescription>
                      {insight.computed_at
                        ? formatDistanceToNow(parseISO(insight.computed_at as string), { addSuffix: true })
                        : ""}
                    </CardDescription>
                  </div>
                  <div className={`flex items-center gap-1 text-sm font-medium ${config.color}`}>
                    <Icon className="h-4 w-4" />
                    {riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)}
                    <span className="ml-1 text-xs opacity-60">({insight.score as number}/100)</span>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-1">
                  <p className="text-xs font-medium text-muted-foreground uppercase">Why this rating</p>
                  <ul className="space-y-1">
                    {reasons.map((reason, i) => (
                      <li key={i} className="text-sm flex items-start gap-2">
                        <span className="text-primary mt-0.5">•</span>
                        {reason}
                      </li>
                    ))}
                  </ul>
                </div>

                {insight.recommendation && (
                  <div className="p-3 rounded-md bg-primary/5 text-sm">
                    <strong>Recommendation:</strong> {insight.recommendation as string}
                  </div>
                )}

                {signals && Object.keys(signals).length > 0 && (
                  <details className="text-xs text-muted-foreground">
                    <summary className="cursor-pointer hover:text-foreground">
                      Raw signals
                    </summary>
                    <pre className="mt-1 p-2 bg-muted rounded text-xs overflow-x-auto">
                      {JSON.stringify(signals, null, 2)}
                    </pre>
                  </details>
                )}
              </CardContent>
            </Card>
          );
        })
      )}
    </div>
  );
}

function formatInsightType(type: string): string {
  return type
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
