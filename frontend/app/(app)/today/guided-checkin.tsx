"use client";

import { useMemo, useState, useTransition } from "react";
import { saveCheckIn, type EntryPayload } from "./actions";
import { RoutineChecklist } from "./routine-checklist";
import { MedChecklist } from "./med-checklist";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { getPlanProgress } from "@/lib/guided-plan";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  Brain,
  CheckCircle2,
  Loader2,
  Moon,
  NotebookPen,
  Sparkles,
  Target,
  Zap,
} from "lucide-react";

type RoutineWithStatus = { id: string; name: string; completed: boolean };
type MedWithStatus = {
  id: string;
  name: string;
  dosage: string | null;
  reminder_time: string | null;
  was_taken: boolean;
  was_skipped: boolean;
};

type Props = {
  initialEntry: Record<string, unknown> | null;
  routines: RoutineWithStatus[];
  meds: MedWithStatus[];
  cards: string[];
  checkInsCompleted: number;
};

const HOURS_OPTIONS = [
  { label: "<4", h: 3.5 },
  { label: "5", h: 5 },
  { label: "6", h: 6 },
  { label: "7", h: 7 },
  { label: "8", h: 8 },
  { label: "9+", h: 9.5 },
];

const ENCOURAGEMENTS = [
  "Nice work showing up today.",
  "Every check-in builds your baseline.",
  "Small, steady tracking adds up.",
  "You're getting to know your patterns.",
  "Consistency is the whole game — well done.",
];

function num(v: unknown): number | null {
  return typeof v === "number" ? v : null;
}

/** A row of quick-select buttons. */
function QuickPick<T extends string | number>({
  options,
  value,
  onChange,
  ariaLabel,
}: {
  options: { label: string; value: T }[];
  value: T | null;
  onChange: (v: T) => void;
  ariaLabel: string;
}) {
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label={ariaLabel}>
      {options.map((o) => {
        const active = value === o.value;
        return (
          <button
            key={String(o.value)}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(o.value)}
            className={cn(
              "min-w-10 rounded-md border px-3 py-2 text-sm font-medium transition-colors",
              active ? "border-primary bg-primary text-primary-foreground" : "border-border hover:bg-muted/60",
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/** A Yes / No toggle pair. */
function YesNo({
  value,
  onChange,
}: {
  value: boolean | null;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex gap-2" role="group">
      {[
        { label: "Yes", v: true },
        { label: "No", v: false },
      ].map(({ label, v }) => {
        const active = value === v;
        return (
          <button
            key={label}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(v)}
            className={cn(
              "flex-1 rounded-md border px-4 py-2.5 text-sm font-medium transition-colors",
              active ? "border-primary bg-primary text-primary-foreground" : "border-border hover:bg-muted/60",
            )}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

function Question({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <p className="text-sm font-medium">{label}</p>
      {children}
    </div>
  );
}

export function GuidedCheckin({ initialEntry, routines, meds, cards, checkInsCompleted }: Props) {
  const d = initialEntry;
  const has = (c: string) => cards.includes(c);

  // Sleep
  const [sleepHours, setSleepHours] = useState<number | null>(
    d?.sleep_minutes != null ? (d.sleep_minutes as number) / 60 : null,
  );
  const [sleepQuality, setSleepQuality] = useState<number | null>(num(d?.sleep_quality));
  // Mood
  const [moodLow, setMoodLow] = useState<boolean | null>(
    d?.mood_valence != null ? (d.mood_valence as number) < 0 : null,
  );
  const [moodValence, setMoodValence] = useState<number | null>(num(d?.mood_valence));
  const [anxietyHigh, setAnxietyHigh] = useState<boolean | null>(
    d?.anxiety != null ? (d.anxiety as number) >= 6 : null,
  );
  const [anxiety, setAnxiety] = useState<number | null>(num(d?.anxiety));
  // Focus
  const [focusGood, setFocusGood] = useState<boolean | null>(
    d?.focus != null ? (d.focus as number) >= 6 : null,
  );
  const [focus, setFocus] = useState<number | null>(num(d?.focus));
  const [productivity, setProductivity] = useState<number | null>(num(d?.productivity));
  // Migraine
  const [migraine, setMigraine] = useState<boolean | null>(
    d?.migraine != null ? (d.migraine as boolean) : null,
  );
  const [migraineAura, setMigraineAura] = useState<boolean | null>(
    d?.migraine_aura != null ? (d.migraine_aura as boolean) : null,
  );
  const [migraineIntensity, setMigraineIntensity] = useState<number | null>(num(d?.migraine_intensity));
  // Journal
  const [reflection, setReflection] = useState<string>((d?.notes as string) ?? "");

  const [isPending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ score: number; day: number } | null>(null);

  const encouragement = useMemo(
    () => ENCOURAGEMENTS[Math.floor(Math.random() * ENCOURAGEMENTS.length)],
    [],
  );

  function handleComplete() {
    setError(null);

    const payload: EntryPayload = {
      // Sleep
      sleep_minutes: has("sleep") && sleepHours != null ? Math.round(sleepHours * 60) : num(d?.sleep_minutes),
      sleep_quality: has("sleep") ? sleepQuality : num(d?.sleep_quality),
      bed_time: (d?.bed_time as string) ?? null,
      wake_time: (d?.wake_time as string) ?? null,
      // Mood
      mood_valence: has("mood") ? moodValence : num(d?.mood_valence),
      anxiety: has("mood") ? anxiety : num(d?.anxiety),
      depression: num(d?.depression),
      mania: num(d?.mania),
      // Focus
      focus: has("focus") ? focus : num(d?.focus),
      productivity: has("focus") ? productivity : num(d?.productivity),
      // Migraine
      migraine: has("migraine") ? migraine === true : (d?.migraine as boolean) ?? false,
      migraine_intensity:
        has("migraine") && migraine ? migraineIntensity : num(d?.migraine_intensity),
      migraine_aura: has("migraine") && migraine ? migraineAura : ((d?.migraine_aura as boolean) ?? null),
      // Journal
      notes: has("journal") ? reflection.trim() || null : ((d?.notes as string) ?? null),
    };

    startTransition(async () => {
      const res = await saveCheckIn(payload);
      if (res?.error) {
        setError(res.error);
        return;
      }
      const day = getPlanProgress(checkInsCompleted + (res?.created ? 1 : 0)).displayDay;
      setResult({ score: res!.score!, day });
    });
  }

  // ---- Completion screen ----
  if (result) {
    const progress = getPlanProgress(checkInsCompleted + 0); // day already computed
    const showDay = result.day;
    const baselineLeft = Math.max(0, 10 - showDay);
    return (
      <Card>
        <CardHeader className="text-center">
          <div className="mb-3 flex justify-center">
            <div className="rounded-full bg-primary/10 p-3">
              <CheckCircle2 className="h-8 w-8 text-primary" />
            </div>
          </div>
          <CardTitle className="text-2xl">Check-in complete</CardTitle>
          <CardDescription>{encouragement}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="flex items-center justify-center gap-8">
            <div className="text-center">
              <p className="text-4xl font-bold text-primary">{result.score}</p>
              <p className="text-xs text-muted-foreground">MindMap Score</p>
            </div>
            <div className="text-center">
              <p className="text-4xl font-bold">{showDay}</p>
              <p className="text-xs text-muted-foreground">
                Day of {progress.phaseTotalDays} · {progress.phaseLabel}
              </p>
            </div>
          </div>
          {baselineLeft > 0 ? (
            <p className="text-center text-sm text-muted-foreground">
              {baselineLeft} more {baselineLeft === 1 ? "day" : "days"} to unlock your Baseline Report.
            </p>
          ) : (
            <p className="text-center text-sm font-medium text-primary">
              Your 10-day baseline is complete — your Baseline Report is ready.
            </p>
          )}
          <MedicalDisclaimer variant="compact" className="text-center" />
          <div className="flex gap-3">
            <Button variant="outline" className="flex-1" onClick={() => setResult(null)}>
              Edit check-in
            </Button>
            <Button className="flex-1" asChild>
              <a href="/home">Go home</a>
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  // ---- Guided form ----
  return (
    <div className="space-y-5">
      {has("sleep") && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Moon className="h-5 w-5 text-primary" /> Sleep
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Question label="Hours of sleep?">
              <QuickPick
                ariaLabel="Hours of sleep"
                options={HOURS_OPTIONS.map((o) => ({ label: o.label, value: o.h }))}
                value={sleepHours}
                onChange={setSleepHours}
              />
            </Question>
            <Question label="Sleep quality?">
              <QuickPick
                ariaLabel="Sleep quality"
                options={[1, 2, 3, 4, 5].map((n) => ({ label: String(n), value: n }))}
                value={sleepQuality}
                onChange={setSleepQuality}
              />
            </Question>
          </CardContent>
        </Card>
      )}

      {has("mood") && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Brain className="h-5 w-5 text-primary" /> Mood
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Question label="Anxiety high today?">
              <YesNo value={anxietyHigh} onChange={(v) => { setAnxietyHigh(v); setAnxiety(v ? 7 : 2); }} />
              {anxietyHigh !== null && (
                <div className="pt-1">
                  <p className="mb-1 text-xs text-muted-foreground">Optional: rate it (0–10)</p>
                  <QuickPick
                    ariaLabel="Anxiety level"
                    options={Array.from({ length: 11 }, (_, i) => ({ label: String(i), value: i }))}
                    value={anxiety}
                    onChange={setAnxiety}
                  />
                </div>
              )}
            </Question>
            <Question label="Mood low today?">
              <YesNo value={moodLow} onChange={(v) => { setMoodLow(v); setMoodValence(v ? -2 : 1); }} />
              {moodLow !== null && (
                <div className="pt-1">
                  <p className="mb-1 text-xs text-muted-foreground">Optional: where's your mood? (−3 low → +3 high)</p>
                  <QuickPick
                    ariaLabel="Mood level"
                    options={[-3, -2, -1, 0, 1, 2, 3].map((n) => ({ label: n > 0 ? `+${n}` : String(n), value: n }))}
                    value={moodValence}
                    onChange={setMoodValence}
                  />
                </div>
              )}
            </Question>
          </CardContent>
        </Card>
      )}

      {has("focus") && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Target className="h-5 w-5 text-primary" /> Focus
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Question label="Focus good today?">
              <YesNo value={focusGood} onChange={(v) => { setFocusGood(v); setFocus(v ? 7 : 3); }} />
              {focusGood !== null && (
                <div className="pt-1">
                  <p className="mb-1 text-xs text-muted-foreground">Optional: rate it (0–10)</p>
                  <QuickPick
                    ariaLabel="Focus level"
                    options={Array.from({ length: 11 }, (_, i) => ({ label: String(i), value: i }))}
                    value={focus}
                    onChange={setFocus}
                  />
                </div>
              )}
            </Question>
            <Question label="Productivity today?">
              <QuickPick
                ariaLabel="Productivity"
                options={[0, 25, 50, 75, 100].map((n) => ({ label: `${n}%`, value: n }))}
                value={productivity}
                onChange={setProductivity}
              />
            </Question>
          </CardContent>
        </Card>
      )}

      {has("migraine") && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Zap className="h-5 w-5 text-primary" /> Migraine / symptoms
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Question label="Migraine today?">
              <YesNo value={migraine} onChange={setMigraine} />
            </Question>
            {migraine && (
              <>
                <Question label="Aura?">
                  <YesNo value={migraineAura} onChange={setMigraineAura} />
                </Question>
                <Question label="Intensity? (0–10)">
                  <QuickPick
                    ariaLabel="Migraine intensity"
                    options={Array.from({ length: 11 }, (_, i) => ({ label: String(i), value: i }))}
                    value={migraineIntensity}
                    onChange={setMigraineIntensity}
                  />
                </Question>
              </>
            )}
          </CardContent>
        </Card>
      )}

      {has("medication") && meds.length > 0 && <MedChecklist meds={meds} />}
      {has("routines") && routines.length > 0 && <RoutineChecklist routines={routines} />}

      {has("journal") && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <NotebookPen className="h-5 w-5 text-primary" /> Reflection
            </CardTitle>
            <CardDescription>Optional — a quick note to yourself.</CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea
              placeholder="Anything on your mind — triggers, wins, observations…"
              value={reflection}
              onChange={(e) => setReflection(e.target.value)}
              rows={3}
            />
          </CardContent>
        </Card>
      )}

      <div className="sticky bottom-4 z-10">
        <Button onClick={handleComplete} disabled={isPending} size="lg" className="w-full shadow-lg">
          {isPending ? <Loader2 className="animate-spin" /> : <Sparkles />}
          {isPending ? "Saving…" : "Complete check-in"}
        </Button>
        {error && <p className="mt-2 text-center text-sm text-destructive">{error}</p>}
      </div>

      <MedicalDisclaimer variant="compact" className="text-center" />
    </div>
  );
}
