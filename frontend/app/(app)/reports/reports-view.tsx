"use client";

import { useState, useTransition } from "react";
import { generateReport, type ReportRow } from "./actions";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText, Loader2, ChevronDown, ChevronUp, Sparkles, Download } from "lucide-react";

const escapeHtml = (s: string) =>
  s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] ?? c,
  );

/**
 * Zero-dependency PDF export: render the report into a print-optimized window
 * and trigger the browser's print dialog (users choose "Save as PDF"). Avoids a
 * server-side PDF toolchain / extra deps.
 */
function saveReportAsPdf(r: ReportRow) {
  const w = window.open("", "_blank", "width=820,height=1040");
  if (!w) return; // popup blocked
  // Every interpolated value below is passed through escapeHtml, so the
  // document.write template cannot be used as an injection vector.
  const title = `MindMap ${r.report_type} report`;
  const insights = (r.key_insights ?? [])
    .map((k) => `<li>${escapeHtml(k)}</li>`)
    .join("");
  w.document.write(`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>${escapeHtml(title)}</title>
<style>
  *{box-sizing:border-box}
  body{font:14px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#111;margin:40px;max-width:720px}
  h1{font-size:20px;margin:0 0 4px}
  .period{color:#555;margin:0 0 20px;font-size:13px}
  .summary{white-space:pre-wrap}
  h2{font-size:14px;margin:24px 0 8px}
  ul{margin:0;padding-left:18px}
  .disclaimer{margin-top:28px;padding-top:12px;border-top:1px solid #ddd;color:#666;font-size:11px}
  @media print{body{margin:0}}
</style></head><body>
  <h1>${escapeHtml(title)}</h1>
  <p class="period">${escapeHtml(r.period_start)} – ${escapeHtml(r.period_end)}</p>
  <div class="summary">${escapeHtml(r.summary_markdown ?? "")}</div>
  ${insights ? `<h2>Key insights</h2><ul>${insights}</ul>` : ""}
  <p class="disclaimer">MindMap is a wellness self-tracking tool, not a medical device. This summary is informational and not a diagnosis. Consult a qualified professional for medical concerns.</p>
</body></html>`);
  w.document.close();
  w.focus();
  // Print once content is laid out; close the tab after the dialog resolves.
  w.onload = () => {
    w.print();
    w.onafterprint = () => w.close();
  };
}

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
                    <div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => saveReportAsPdf(r)}
                      >
                        <Download /> Save as PDF
                      </Button>
                    </div>
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
