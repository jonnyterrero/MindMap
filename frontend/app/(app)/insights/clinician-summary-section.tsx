import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import {
  AlertTriangle,
  CalendarCheck,
  FileText,
  Minus,
  Quote,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import type { ClinicianSummaryRow, SummaryCrisis } from "./summary-actions";

function DirectionIcon({ direction }: { direction: string }) {
  if (direction === "rising") return <TrendingUp className="h-4 w-4 text-amber-500" />;
  if (direction === "falling") return <TrendingDown className="h-4 w-4 text-amber-500" />;
  return <Minus className="h-4 w-4 text-muted-foreground" />;
}

function CrisisBanner({ crisis }: { crisis: SummaryCrisis }) {
  return (
    <div className="rounded-lg border border-destructive/50 bg-destructive/5 p-4">
      <div className="flex items-center gap-2 font-medium text-destructive">
        <AlertTriangle className="h-5 w-5" /> {crisis.title}
      </div>
      <p className="mt-1 text-sm text-muted-foreground">{crisis.body}</p>
      <ul className="mt-2 space-y-1 text-sm">
        {crisis.resources.map((r) => (
          <li key={r.label}>
            {r.href ? (
              <a href={r.href} className="font-medium underline">
                {r.label}
              </a>
            ) : (
              <span className="font-medium">{r.label}</span>
            )}
            <span className="text-muted-foreground"> — {r.detail}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function ClinicianSummarySection({ summary }: { summary: ClinicianSummaryRow | null }) {
  if (!summary) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-lg">
            <FileText className="h-5 w-5 text-primary" /> Patterns &amp; Clinician Summary
          </CardTitle>
          <CardDescription>A shareable, evidence-based summary of your own data. Not a diagnosis.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
            Your summary will appear here once the daily analysis has run on a few days of check-ins.
          </div>
        </CardContent>
      </Card>
    );
  }

  const p = summary.payload;
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <FileText className="h-5 w-5 text-primary" /> Patterns &amp; Clinician Summary
        </CardTitle>
        <CardDescription>
          {p.date_range.length === 2
            ? `Covering ${p.date_range[0]} – ${p.date_range[1]}. `
            : ""}
          A shareable summary of your own data. Not a diagnosis.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {p.crisis && <CrisisBanner crisis={p.crisis} />}

        {/* Data completeness */}
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <CalendarCheck className="h-4 w-4" />
          {p.completeness.logged_days} days logged · {Math.round(p.completeness.adherence * 100)}% adherence ·
          longest streak {p.completeness.longest_streak}
        </div>

        {p.abstained ? (
          <div className="rounded-lg border border-dashed p-6 text-center">
            <p className="text-sm font-medium">Not enough data yet to surface reliable patterns.</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Keep logging — about {p.readiness.days_remaining} more day
              {p.readiness.days_remaining === 1 ? "" : "s"} (target {p.readiness.recommended_min_days}).
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {p.trajectories.length > 0 && (
              <section>
                <h3 className="mb-2 text-sm font-semibold">Trajectories</h3>
                <ul className="space-y-1 text-sm">
                  {p.trajectories.map((t) => (
                    <li key={t.metric} className="flex items-center gap-2">
                      <DirectionIcon direction={t.direction} />
                      <span>{t.statement}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {p.detected_patterns.length > 0 && (
              <section>
                <h3 className="mb-2 text-sm font-semibold">Detected patterns</h3>
                <ul className="space-y-2 text-sm">
                  {p.detected_patterns.map((d, i) => (
                    <li key={i} className="rounded-lg border p-3">
                      <p>{d.statement}</p>
                      {d.citations.length > 0 && (
                        <p className="mt-1 flex items-start gap-1 text-xs text-muted-foreground">
                          <Quote className="mt-0.5 h-3 w-3 shrink-0" />
                          {d.citations.join("; ")}
                        </p>
                      )}
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {p.watch_items.length > 0 && (
              <section>
                <h3 className="mb-2 text-sm font-semibold">Worth watching</h3>
                <ul className="space-y-1 text-sm">
                  {p.watch_items.map((w, i) => (
                    <li key={i}>{w.statement}</li>
                  ))}
                </ul>
              </section>
            )}

            {(p.instruments.phq9 || p.instruments.gad7) && (
              <section>
                <h3 className="mb-2 text-sm font-semibold">Screening (not a diagnosis)</h3>
                <ul className="space-y-1 text-sm">
                  {p.instruments.phq9 && (
                    <li>PHQ-9: {p.instruments.phq9.total} ({p.instruments.phq9.severity})</li>
                  )}
                  {p.instruments.gad7 && (
                    <li>GAD-7: {p.instruments.gad7.total} ({p.instruments.gad7.severity})</li>
                  )}
                </ul>
              </section>
            )}
          </div>
        )}

        <MedicalDisclaimer variant="compact" />
      </CardContent>
    </Card>
  );
}
