import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { ChevronRight, FileText, Activity } from "lucide-react";
import type { PatientSummary } from "@/app/(app)/provider/actions";

const RISK_STYLE: Record<string, string> = {
  low: "bg-green-500/15 text-green-600 dark:text-green-400",
  moderate: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  high: "bg-orange-500/15 text-orange-600 dark:text-orange-400",
  critical: "bg-red-500/15 text-red-600 dark:text-red-400",
};

export function PatientSummaryCard({ patient }: { patient: PatientSummary }) {
  return (
    <Link href={`/provider/${patient.patientId}`} className="block">
      <Card className="transition-colors hover:bg-muted/50">
        <CardContent className="flex items-center justify-between gap-3 p-4">
          <div className="min-w-0">
            <p className="truncate font-medium">{patient.name}</p>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              {patient.latestRiskLevel ? (
                <span className={cn("inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-medium", RISK_STYLE[patient.latestRiskLevel] ?? "")}>
                  <Activity className="h-3 w-3" /> {patient.latestRiskLevel} risk
                </span>
              ) : (
                <span className="text-muted-foreground">No predictions yet</span>
              )}
              {patient.lastReportLabel && (
                <span className="inline-flex items-center gap-1">
                  <FileText className="h-3 w-3" /> {patient.lastReportLabel}
                </span>
              )}
            </div>
          </div>
          <ChevronRight className="h-5 w-5 shrink-0 text-muted-foreground" />
        </CardContent>
      </Card>
    </Link>
  );
}
