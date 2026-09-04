"use client";

import { useState, useTransition } from "react";
import {
  generateInsights,
  getInsightHistory,
  submitInsightFeedback,
  type InsightHistoryPoint,
} from "./actions";
import { Button } from "@/components/ui/button";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import {
  RefreshCw, Loader2, AlertTriangle, TrendingUp, Brain, CheckCircle2, Info,
  ThumbsUp, ThumbsDown, History,
} from "lucide-react";
import { format, formatDistanceToNow, parseISO } from "date-fns";
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

export function InsightsView({
  insights: initialInsights,
  initialFeedback = {},
}: {
  insights: Insight[];
  initialFeedback?: Record<string, boolean>;
}) {
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
        insights.map((insight) => (
          <InsightCard
            key={insight.id as string}
            insight={insight}
            initialVote={initialFeedback[insight.id as string] ?? null}
          />
        ))
      )}
    </div>
  );
}

function InsightCard({
  insight,
  initialVote,
}: {
  insight: Insight;
  initialVote: boolean | null;
}) {
  const riskLevel = (insight.risk_level as string) ?? "unknown";
  const config = RISK_CONFIG[riskLevel] ?? RISK_CONFIG.unknown;
  const Icon = config.icon;
  const reasons = (insight.reasons as string[]) ?? [];
  const signals = insight.signals as Record<string, unknown> | null;
  const insightType = insight.insight_type as string;

  const [vote, setVote] = useState<boolean | null>(initialVote);
  const [, startVote] = useTransition();
  const [history, setHistory] = useState<InsightHistoryPoint[] | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);

  function handleVote(helpful: boolean) {
    // Optimistic; re-voting just overwrites.
    setVote(helpful);
    startVote(async () => {
      await submitInsightFeedback(insight.id as string, insightType, helpful);
    });
  }

  async function toggleHistory() {
    const next = !historyOpen;
    setHistoryOpen(next);
    if (next && history === null && !historyLoading) {
      setHistoryLoading(true);
      try {
        setHistory(await getInsightHistory(insightType));
      } finally {
        setHistoryLoading(false);
      }
    }
  }

  return (
    <Card className="glass-card">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-base flex items-center gap-2">
              {insightType === "migraine_risk" ? (
                <AlertTriangle className="h-4 w-4 text-primary" />
              ) : (
                <TrendingUp className="h-4 w-4 text-primary" />
              )}
              {formatInsightType(insightType)}
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

        {(insight.recommendation as string) && (
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

        <div className="flex items-center justify-between border-t pt-3">
          <button
            type="button"
            onClick={toggleHistory}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            aria-expanded={historyOpen}
          >
            <History className="h-3.5 w-3.5" />
            {historyOpen ? "Hide history" : "History"}
          </button>

          <div className="flex items-center gap-1.5">
            <span className="text-xs text-muted-foreground mr-1">
              {vote === null ? "Helpful?" : "Thanks for the feedback"}
            </span>
            <Button
              variant={vote === true ? "secondary" : "ghost"}
              size="icon"
              className="h-7 w-7"
              aria-label="Mark insight helpful"
              aria-pressed={vote === true}
              onClick={() => handleVote(true)}
            >
              <ThumbsUp className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant={vote === false ? "secondary" : "ghost"}
              size="icon"
              className="h-7 w-7"
              aria-label="Mark insight not helpful"
              aria-pressed={vote === false}
              onClick={() => handleVote(false)}
            >
              <ThumbsDown className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>

        {historyOpen && (
          <div className="rounded-md bg-muted/40 p-3">
            {historyLoading ? (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading history…
              </div>
            ) : !history || history.length < 2 ? (
              <p className="text-xs text-muted-foreground">
                Not enough history yet — refresh insights on different days to build a trend.
              </p>
            ) : (
              <InsightHistoryChart points={history} />
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/** Tiny dependency-free bar sparkline: score 0-100 over time, oldest → newest. */
function InsightHistoryChart({ points }: { points: InsightHistoryPoint[] }) {
  const chronological = [...points].reverse();
  return (
    <div className="space-y-2">
      <p className="text-xs font-medium text-muted-foreground uppercase">
        Score over time
      </p>
      <div className="flex items-end gap-1 h-16">
        {chronological.map((p, i) => {
          const score = p.score ?? 0;
          const config = RISK_CONFIG[p.risk_level] ?? RISK_CONFIG.unknown;
          return (
            <div
              key={i}
              className="group relative flex-1 min-w-[4px] max-w-[24px]"
              title={`${format(parseISO(p.computed_at), "MMM d")}: ${score}/100 (${p.risk_level})`}
            >
              <div
                className={`w-full rounded-sm bg-current opacity-70 group-hover:opacity-100 transition-opacity ${config.color}`}
                style={{ height: `${Math.max(6, (score / 100) * 64)}px` }}
              />
            </div>
          );
        })}
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>{format(parseISO(chronological[0].computed_at), "MMM d")}</span>
        <span>{format(parseISO(chronological[chronological.length - 1].computed_at), "MMM d")}</span>
      </div>
    </div>
  );
}

function formatInsightType(type: string): string {
  return type
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
