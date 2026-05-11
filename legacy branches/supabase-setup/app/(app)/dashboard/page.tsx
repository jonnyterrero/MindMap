import { getLast30DaysEntries, getMigraineRiskToday } from "./actions";
import { DashboardCharts } from "./dashboard-charts";
import { MigraineRiskCard } from "./migraine-risk-card";

export default async function DashboardPage() {
  const [entries, migraineRisk] = await Promise.all([
    getLast30DaysEntries(),
    getMigraineRiskToday(),
  ]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Your mental health trends over the last 30 days
        </p>
      </div>

      {migraineRisk && <MigraineRiskCard risk={migraineRisk} />}

      {entries.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          <p className="text-lg font-medium">No data yet</p>
          <p>Start logging on your Today page to see trends here.</p>
        </div>
      ) : (
        <DashboardCharts entries={entries} />
      )}
    </div>
  );
}
