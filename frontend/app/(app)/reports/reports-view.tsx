"use client";

import { useState, useTransition } from "react";
import { generateReport, type ReportRow } from "./actions";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText, Loader2, ChevronDown, ChevronUp, Sparkles } from "lucide-react";

export function ReportsView({ initial }: { initial: ReportRow[] }) {
  const [reports, setReports] = useState<ReportRow[]>(initial);
  const [openId, setOpenId] = useState<string | null>(initial[0]?.id ?? null);
  const [pendingType, setPendingType] = useState<"weekly" | "monthly" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [, startTransition] = useTransition();

  function generate(type: "weekly" | "monthly") {
    setError(null);
    setPendingType(type);
    startTransition(async () => {
      const r = await generateReport(type);
      setPendingType(null);
      if ("error" in r) {
        setError(r.error);
      } else {
        setReports((prev) => [r.report, ...prev.filter((x) => x.id !== r.report.id)]);
        setOpenId(r.report.id);
      }
    });
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-2">
        <Button onClick={() => generate("weekly")} disabled={pendingType !== null}>
          {pendingType === "weekly" ? <Loader2 className="animate-spin" /> : <Sparkles />}
          Generate weekly
        </Button>
        <Button variant="outline" onClick={() => generate("monthly")} disabled={pendingType !== null}>
          {pendingType === "monthly" ? <Loader2 className="animate-spin" /> : <Sparkles />}
          Generate monthly
        </Button>
      </div>
      {error && <p className="text-sm text-destructive">{error}</p>}

      {reports.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 py-10 text-center">
            <FileText className="h-8 w-8 text-muted-foreground opacity-40" />
            <p className="text-sm text-muted-foreground">
              No reports yet. Generate one above — it summarizes your recent patterns.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {reports.map((r) => {
            const open = openId === r.id;
            return (
              <Card key={r.id}>
                <CardHeader className="pb-2">
                  <button
                    type="button"
                    onClick={() => setOpenId(open ? null : r.id)}
                    className="flex w-full items-center justify-between gap-2 text-left"
                  >
                    <CardTitle className="text-base capitalize">
                      {r.report_type} report · {r.period_start} – {r.period_end}
                    </CardTitle>
                    {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </button>
                </CardHeader>
                {open && (
                  <CardContent className="space-y-3">
                    {r.summary_markdown && (
                      <div className="whitespace-pre-wrap text-sm leading-relaxed text-foreground/90">
                        {r.summary_markdown}
                      </div>
                    )}
                    {r.key_insights?.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {r.key_insights.map((k, i) => (
                          <span key={i} className="rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary">
                            {k}
                          </span>
                        ))}
                      </div>
                    )}
                    <MedicalDisclaimer variant="inline" />
                  </CardContent>
                )}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
