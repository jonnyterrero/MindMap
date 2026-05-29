import { getLatestInsights, getCorrelations } from "./actions";
import { InsightsView } from "./insights-view";
import { CorrelationsCard } from "./correlations-card";

export default async function InsightsPage() {
  const [insights, correlations] = await Promise.all([
    getLatestInsights(),
    getCorrelations(),
  ]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Insights</h1>
        <p className="text-muted-foreground">
          Patterns from your own check-ins. Every insight shows why it appeared —
          and none of it is medical advice.
        </p>
      </div>
      <CorrelationsCard correlations={correlations} />
      <InsightsView insights={insights} />
    </div>
  );
}
