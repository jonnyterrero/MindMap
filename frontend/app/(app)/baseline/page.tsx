import Link from "next/link";
import { getBaselineData } from "./actions";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { BASELINE_DAYS } from "@/lib/guided-plan";
import { ArrowLeft, Moon, Brain, Zap, Target, ListChecks, Pill, Sparkles } from "lucide-react";

function Stat({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Moon;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-border p-4">
      <div className="mb-1 flex items-center gap-1.5 text-xs text-muted-foreground">
        <Icon className="h-3.5 w-3.5" /> {label}
      </div>
      <p className="text-xl font-semibold">{value}</p>
    </div>
  );
}

export default async function BaselinePage() {
  const data = await getBaselineData();

  if (!data.unlocked) {
    const done = data.checkInsCompleted;
    return (
      <div className="mx-auto max-w-lg space-y-5">
        <Card>
          <CardHeader className="text-center">
            <div className="mb-3 flex justify-center">
              <div className="rounded-full bg-primary/10 p-3">
                <Sparkles className="h-8 w-8 text-primary" />
              </div>
            </div>
            <CardTitle className="text-2xl">Your Baseline Report is on the way</CardTitle>
            <CardDescription>
              Complete {BASELINE_DAYS} daily check-ins to unlock your first report. You&apos;ve
              done {done} so far.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Progress value={(done / BASELINE_DAYS) * 100} className="h-2" />
            <p className="text-center text-sm text-muted-foreground">
              {data.remaining} more {data.remaining === 1 ? "day" : "days"} to go.
            </p>
            <Button asChild className="w-full">
              <Link href="/today">Do today&apos;s check-in</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const fmt = (v: number | null, suffix = "", digits = 1) =>
    v == null ? "—" : `${v.toFixed(digits)}${suffix}`;
  const pct = (v: number | null) => (v == null ? "—" : `${v}%`);

  return (
    <div className="mx-auto max-w-lg space-y-5">
      <div>
        <Link
          href="/"
          className="mb-2 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Home
        </Link>
        <h1 className="text-2xl font-bold tracking-tight">Your Baseline Report</h1>
        <p className="text-muted-foreground">
          A summary of your first {data.checkInsCompleted} check-ins.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Stat icon={Moon} label="Avg sleep" value={fmt(data.avgSleepHours, "h")} />
        <Stat icon={Moon} label="Avg sleep quality" value={fmt(data.avgSleepQuality, "/5")} />
        <Stat icon={Zap} label="Migraine days" value={String(data.migraineDays)} />
        <Stat icon={Zap} label="Avg intensity" value={fmt(data.avgMigraineIntensity, "/10")} />
        <Stat icon={Brain} label="Avg anxiety" value={fmt(data.avgAnxiety, "/10")} />
        <Stat icon={Target} label="Avg focus" value={fmt(data.avgFocus, "/10")} />
        <Stat icon={ListChecks} label="Routine completion" value={pct(data.routineCompletionPct)} />
        <Stat icon={Pill} label="Medication consistency" value={pct(data.medicationConsistencyPct)} />
      </div>

      {data.pattern && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">A possible pattern</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm">{data.pattern}</p>
            <MedicalDisclaimer variant="inline" />
          </CardContent>
        </Card>
      )}

      <MedicalDisclaimer variant="full" />
    </div>
  );
}
