import Link from "next/link";
import { getPatientSummary } from "../actions";
import { CorrelationsCard } from "@/app/(app)/insights/correlations-card";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ArrowLeft } from "lucide-react";

const RISK_STYLE: Record<string, string> = {
  low: "bg-green-500/15 text-green-600 dark:text-green-400",
  moderate: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  high: "bg-orange-500/15 text-orange-600 dark:text-orange-400",
  critical: "bg-red-500/15 text-red-600 dark:text-red-400",
};
const TYPE_LABEL: Record<string, string> = {
  migraine: "Migraine",
  anxiety: "Anxiety",
  mood: "Mood dip",
  pain_flare: "Pain flare",
};

export default async function PatientDetailPage({
  params,
}: {
  params: Promise<{ patientId: string }>;
}) {
  const { patientId } = await params;
  const result = await getPatientSummary(patientId);

  if ("error" in result) {
    return (
      <div className="mx-auto max-w-lg space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>Access unavailable</CardTitle>
            <CardDescription>{result.error}</CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild variant="outline"><Link href="/provider"><ArrowLeft /> Back</Link></Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const { name, permissions, predictions, reports, correlations } = result;

  return (
    <div className="space-y-5">
      <div>
        <Link href="/provider" className="mb-2 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-3.5 w-3.5" /> Patients
        </Link>
        <h1 className="text-2xl font-bold tracking-tight">{name}</h1>
        <p className="text-muted-foreground">Read-only — shared by the patient.</p>
      </div>

      <Tabs defaultValue="summary">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="summary">Summary</TabsTrigger>
          <TabsTrigger value="predictions">Predictions</TabsTrigger>
          <TabsTrigger value="reports">Reports</TabsTrigger>
          <TabsTrigger value="correlations">Patterns</TabsTrigger>
        </TabsList>

        <TabsContent value="summary" className="space-y-3 pt-3">
          <Card>
            <CardContent className="grid grid-cols-2 gap-3 p-4 text-sm">
              <div>
                <p className="text-xs text-muted-foreground">Predictions shared</p>
                <p className="font-medium">{permissions.read_predictions ? `${predictions.length} active` : "Not shared"}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Reports shared</p>
                <p className="font-medium">{permissions.read_reports ? `${reports.length}` : "Not shared"}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Entry patterns</p>
                <p className="font-medium">{permissions.read_entries ? `${correlations.length} found` : "Not shared"}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Top risk</p>
                <p className="font-medium capitalize">{predictions[0]?.risk_level ?? "—"}</p>
              </div>
            </CardContent>
          </Card>
          <MedicalDisclaimer variant="inline" />
        </TabsContent>

        <TabsContent value="predictions" className="space-y-2 pt-3">
          {!permissions.read_predictions ? (
            <EmptyTab text="The patient hasn't shared predictions." />
          ) : predictions.length === 0 ? (
            <EmptyTab text="No predictions recorded yet." />
          ) : (
            predictions.map((p) => (
              <Card key={p.prediction_type}>
                <CardContent className="flex items-center justify-between p-3">
                  <span className="text-sm font-medium">{TYPE_LABEL[p.prediction_type] ?? p.prediction_type}</span>
                  <span className={cn("rounded-full px-2 py-0.5 text-xs font-semibold capitalize", RISK_STYLE[p.risk_level] ?? "")}>
                    {p.risk_level}
                  </span>
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>

        <TabsContent value="reports" className="space-y-2 pt-3">
          {!permissions.read_reports ? (
            <EmptyTab text="The patient hasn't shared reports." />
          ) : reports.length === 0 ? (
            <EmptyTab text="No reports generated yet." />
          ) : (
            reports.map((r) => (
              <Card key={r.id}>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm capitalize">
                    {r.report_type} report · {r.period_start} – {r.period_end}
                  </CardTitle>
                </CardHeader>
                {r.summary_markdown && (
                  <CardContent>
                    <p className="whitespace-pre-wrap text-sm text-muted-foreground">{r.summary_markdown}</p>
                  </CardContent>
                )}
              </Card>
            ))
          )}
        </TabsContent>

        <TabsContent value="correlations" className="space-y-2 pt-3">
          {!permissions.read_entries ? (
            <EmptyTab text="The patient hasn't shared entry-level data." />
          ) : correlations.length === 0 ? (
            <EmptyTab text="Not enough shared entries to surface patterns yet." />
          ) : (
            <CorrelationsCard correlations={correlations} />
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

function EmptyTab({ text }: { text: string }) {
  return (
    <div className="rounded-lg border border-dashed p-6 text-center">
      <p className="text-sm text-muted-foreground">{text}</p>
    </div>
  );
}
