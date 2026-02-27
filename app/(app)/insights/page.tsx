import { getLatestInsights } from "./actions";
import { InsightsView } from "./insights-view";

export default async function InsightsPage() {
  const insights = await getLatestInsights();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Insights</h1>
        <p className="text-muted-foreground">
          AI-powered analysis of your mental health patterns.
          Every insight shows exactly why it was generated.
        </p>
      </div>
      <InsightsView insights={insights} />
    </div>
  );
}
