import { getLatestInsights, getCorrelations } from "./actions";
import { getLatestPredictions } from "./prediction-actions";
import { getLatestClinicianSummary } from "./summary-actions";
import { InsightsView } from "./insights-view";
import { CorrelationsCard } from "./correlations-card";
import { PredictionsSection } from "./predictions-section";
import { ClinicianSummarySection } from "./clinician-summary-section";

export default async function InsightsPage() {
  const [insights, correlations, predictions, summary] = await Promise.all([
    getLatestInsights(),
    getCorrelations(),
    getLatestPredictions(),
    getLatestClinicianSummary(),
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
      <ClinicianSummarySection summary={summary} />
      <PredictionsSection initial={predictions} />
      <CorrelationsCard correlations={correlations} />
      <InsightsView insights={insights} />
    </div>
  );
}
