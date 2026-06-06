import { getReports } from "./actions";
import { ReportsView } from "./reports-view";

export default async function ReportsPage() {
  const reports = await getReports();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Reports</h1>
        <p className="text-muted-foreground">
          AI-written summaries of your weekly and monthly patterns — from your own data, not medical advice.
        </p>
      </div>
      <ReportsView initial={reports} />
    </div>
  );
}
