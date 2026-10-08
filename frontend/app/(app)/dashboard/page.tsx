import Link from "next/link";
import { getLast30DaysEntries, getMigraineRiskToday } from "./actions";
import { DashboardCharts } from "./dashboard-charts";
import { MigraineRiskCard } from "./migraine-risk-card";
import { Button } from "@/components/ui/button";

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
        <div className="py-12 text-center text-muted-foreground">
          <p className="text-lg font-medium text-foreground">No data yet</p>
          <p className="mb-4">Complete a check-in and your 30-day trends will show up here.</p>
          <Button asChild>
            <Link href="/today">Start check-in</Link>
          </Button>
        </div>
      ) : (
        <DashboardCharts entries={entries} />
      )}
    </div>
  );
}
