"use client";

import { useState, useTransition } from "react";
import { upsertTodayEntry, type EntryPayload } from "./actions";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Moon,
  Brain,
  Zap,
  Save,
  Loader2,
  Check,
  AlertTriangle,
} from "lucide-react";

type Props = {
  initialData: Record<string, unknown> | null;
};

const MOOD_LABELS: Record<number, string> = {
  [-3]: "Very Low",
  [-2]: "Low",
  [-1]: "Slightly Low",
  0: "Neutral",
  1: "Slightly High",
  2: "High",
  3: "Very High",
};

export function TodayForm({ initialData }: Props) {
  const [isPending, startTransition] = useTransition();
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const d = initialData;

  const [sleepHours, setSleepHours] = useState(
    d?.sleep_minutes ? (d.sleep_minutes as number) / 60 : 7
  );
  const [sleepQuality, setSleepQuality] = useState(
    (d?.sleep_quality as number) ?? 3
  );
  const [bedTime, setBedTime] = useState((d?.bed_time as string) ?? "22:00");
  const [wakeTime, setWakeTime] = useState((d?.wake_time as string) ?? "06:00");
  const [moodValence, setMoodValence] = useState(
    (d?.mood_valence as number) ?? 0
  );
  const [anxiety, setAnxiety] = useState((d?.anxiety as number) ?? 0);
  const [depression, setDepression] = useState(
    (d?.depression as number) ?? 0
  );
  const [mania, setMania] = useState((d?.mania as number) ?? 0);
  const [focus, setFocus] = useState((d?.focus as number) ?? 5);
  const [productivity, setProductivity] = useState(
    (d?.productivity as number) ?? 50
  );
  const [migraine, setMigraine] = useState((d?.migraine as boolean) ?? false);
  const [migraineIntensity, setMigraineIntensity] = useState(
    (d?.migraine_intensity as number) ?? 0
  );
  const [migraineAura, setMigraineAura] = useState(
    (d?.migraine_aura as boolean) ?? false
  );
  const [notes, setNotes] = useState((d?.notes as string) ?? "");

  function handleSave() {
    setSaved(false);
    setError(null);

    const payload: EntryPayload = {
      sleep_minutes: Math.round(sleepHours * 60),
      sleep_quality: sleepQuality,
      bed_time: bedTime || null,
      wake_time: wakeTime || null,
      mood_valence: moodValence,
      anxiety,
      depression,
      mania,
      focus,
      productivity,
      migraine,
      migraine_intensity: migraine ? migraineIntensity : null,
      migraine_aura: migraine ? migraineAura : null,
      notes: notes.trim() || null,
    };

    startTransition(async () => {
      const result = await upsertTodayEntry(payload);
      if (result.error) {
        setError(result.error);
      } else {
        setSaved(true);
        setTimeout(() => setSaved(false), 3000);
      }
    });
  }

  return (
    <div className="space-y-6">
      {/* Sleep */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Moon className="h-5 w-5 text-primary" /> Sleep
          </CardTitle>
          <CardDescription>How did you sleep last night?</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Hours slept</Label>
              <span className="text-sm font-medium">{sleepHours.toFixed(1)}h</span>
            </div>
            <Slider
              min={0}
              max={14}
              step={0.5}
              value={[sleepHours]}
              onValueChange={([v]) => setSleepHours(v)}
            />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Sleep quality</Label>
              <span className="text-sm font-medium">{sleepQuality}/5</span>
            </div>
            <Slider
              min={1}
              max={5}
              step={1}
              value={[sleepQuality]}
              onValueChange={([v]) => setSleepQuality(v)}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="bedTime">Bed time</Label>
              <Input
                id="bedTime"
                type="time"
                value={bedTime}
                onChange={(e) => setBedTime(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="wakeTime">Wake time</Label>
              <Input
                id="wakeTime"
                type="time"
                value={wakeTime}
                onChange={(e) => setWakeTime(e.target.value)}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Mood & Mental */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" /> Mood & Mental
          </CardTitle>
          <CardDescription>Rate how you&apos;re feeling</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Mood</Label>
              <span className="text-sm font-medium">
                {MOOD_LABELS[moodValence] ?? moodValence}
              </span>
            </div>
            <Slider
              min={-3}
              max={3}
              step={1}
              value={[moodValence]}
              onValueChange={([v]) => setMoodValence(v)}
            />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Anxiety</Label>
              <span className="text-sm font-medium">{anxiety}/10</span>
            </div>
            <Slider
              min={0}
              max={10}
              step={1}
              value={[anxiety]}
              onValueChange={([v]) => setAnxiety(v)}
            />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Depression</Label>
              <span className="text-sm font-medium">{depression}/10</span>
            </div>
            <Slider
              min={0}
              max={10}
              step={1}
              value={[depression]}
              onValueChange={([v]) => setDepression(v)}
            />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Mania</Label>
              <span className="text-sm font-medium">{mania}/10</span>
            </div>
            <Slider
              min={0}
              max={10}
              step={1}
              value={[mania]}
              onValueChange={([v]) => setMania(v)}
            />
          </div>
        </CardContent>
      </Card>

      {/* Focus & Productivity */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" /> Focus & Productivity
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Focus</Label>
              <span className="text-sm font-medium">{focus}/10</span>
            </div>
            <Slider
              min={0}
              max={10}
              step={1}
              value={[focus]}
              onValueChange={([v]) => setFocus(v)}
            />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Productivity</Label>
              <span className="text-sm font-medium">{productivity}%</span>
            </div>
            <Slider
              min={0}
              max={100}
              step={5}
              value={[productivity]}
              onValueChange={([v]) => setProductivity(v)}
            />
          </div>
        </CardContent>
      </Card>

      {/* Migraine */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-primary" /> Migraine
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="flex items-center justify-between">
            <Label htmlFor="migraine">Migraine today?</Label>
            <Switch
              id="migraine"
              checked={migraine}
              onCheckedChange={setMigraine}
            />
          </div>

          {migraine && (
            <>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Intensity</Label>
                  <span className="text-sm font-medium">
                    {migraineIntensity}/10
                  </span>
                </div>
                <Slider
                  min={0}
                  max={10}
                  step={1}
                  value={[migraineIntensity]}
                  onValueChange={([v]) => setMigraineIntensity(v)}
                />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="aura">Aura present?</Label>
                <Switch
                  id="aura"
                  checked={migraineAura}
                  onCheckedChange={setMigraineAura}
                />
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Notes */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle>Notes</CardTitle>
          <CardDescription>Anything else you want to capture</CardDescription>
        </CardHeader>
        <CardContent>
          <Textarea
            placeholder="How was your day? Any triggers, wins, or observations..."
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={4}
          />
        </CardContent>
      </Card>

      {/* Save */}
      <div className="flex items-center gap-3">
        <Button
          onClick={handleSave}
          disabled={isPending}
          size="lg"
          className="w-full sm:w-auto"
        >
          {isPending ? (
            <Loader2 className="animate-spin" />
          ) : saved ? (
            <Check />
          ) : (
            <Save />
          )}
          {isPending ? "Saving..." : saved ? "Saved!" : "Save Entry"}
        </Button>

        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}
        {saved && (
          <p className="text-sm text-green-600">Entry saved to database.</p>
        )}
      </div>
    </div>
  );
}
