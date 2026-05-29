"use client";

import { useState, useTransition } from "react";
import { completeOnboarding } from "./actions";
import type { FocusOption } from "./constants";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import {
  Activity,
  Brain,
  CheckCircle2,
  Loader2,
  Moon,
  Pill,
  Sparkles,
  Target,
  Wind,
  Zap,
} from "lucide-react";

const FOCUS_CHOICES: { id: FocusOption; label: string; icon: typeof Brain }[] = [
  { id: "migraine", label: "Migraine patterns", icon: Zap },
  { id: "anxiety", label: "Anxiety / stress patterns", icon: Wind },
  { id: "adhd", label: "ADHD / focus patterns", icon: Target },
  { id: "mood", label: "Mood stability", icon: Activity },
  { id: "sleep", label: "Sleep improvement", icon: Moon },
  { id: "medication", label: "Medication consistency", icon: Pill },
];

const CARD_CHOICES: { id: string; label: string; description: string }[] = [
  { id: "sleep", label: "Sleep", description: "Hours and quality" },
  { id: "mood", label: "Mood / anxiety", description: "How you're feeling" },
  { id: "focus", label: "Focus / productivity", description: "Attention and output" },
  { id: "migraine", label: "Migraine / symptoms", description: "Headaches and body signals" },
  { id: "medication", label: "Medication", description: "What you took" },
  { id: "routines", label: "Routines", description: "Habits you're building" },
  { id: "journal", label: "Journal / reflection", description: "A quick note to yourself" },
];

const TOTAL_STEPS = 5;

export function OnboardingFlow() {
  const [step, setStep] = useState(0);
  const [acknowledged, setAcknowledged] = useState(false);
  const [focus, setFocus] = useState<FocusOption | null>(null);
  const [cards, setCards] = useState<Record<string, boolean>>(
    Object.fromEntries(CARD_CHOICES.map((c) => [c.id, true])),
  );
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  const selectedCards = CARD_CHOICES.filter((c) => cards[c.id]).map((c) => c.id);

  const canContinue =
    (step === 1 && !acknowledged) || (step === 2 && !focus) || (step === 3 && selectedCards.length === 0)
      ? false
      : true;

  function next() {
    setError(null);
    setStep((s) => Math.min(s + 1, TOTAL_STEPS - 1));
  }

  function back() {
    setError(null);
    setStep((s) => Math.max(s - 1, 0));
  }

  function finish() {
    if (!focus) return;
    setError(null);
    startTransition(async () => {
      const result = await completeOnboarding({ focus, cards: selectedCards });
      if (result?.error) setError(result.error);
    });
  }

  return (
    <div className="flex flex-1 flex-col gap-6">
      {/* Progress */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5 font-medium">
            <Brain className="h-3.5 w-3.5 text-primary" /> MindMap
          </span>
          <span>
            Step {step + 1} of {TOTAL_STEPS}
          </span>
        </div>
        <Progress value={((step + 1) / TOTAL_STEPS) * 100} className="h-1.5" />
      </div>

      <div className="flex flex-1 flex-col justify-center">
        {step === 0 && (
          <Card>
            <CardHeader className="text-center">
              <div className="mb-3 flex justify-center">
                <div className="rounded-full bg-primary/10 p-3">
                  <Brain className="h-8 w-8 text-primary" />
                </div>
              </div>
              <CardTitle className="text-2xl">Welcome to MindMap</CardTitle>
              <CardDescription className="mx-auto max-w-sm text-balance">
                Track sleep, mood, focus, routines, medications, migraines, and symptoms
                to help you notice personal patterns over time.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-center text-sm text-muted-foreground">
                It takes under 90 seconds a day. Let&apos;s set it up.
              </p>
            </CardContent>
          </Card>
        )}

        {step === 1 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-xl">Before we start</CardTitle>
              <CardDescription>Please read and acknowledge this.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <MedicalDisclaimer variant="full" />
              <label className="flex cursor-pointer items-start gap-3 rounded-lg p-2 hover:bg-muted/50">
                <Checkbox
                  checked={acknowledged}
                  onCheckedChange={(v) => setAcknowledged(!!v)}
                  className="mt-0.5"
                />
                <span className="text-sm">
                  I understand MindMap does not diagnose, treat, cure, or prevent disease,
                  and does not replace medical care.
                </span>
              </label>
            </CardContent>
          </Card>
        )}

        {step === 2 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-xl">What brings you here?</CardTitle>
              <CardDescription>
                Choose your main focus. You can track everything either way — this just
                tailors your experience.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {FOCUS_CHOICES.map(({ id, label, icon: Icon }) => {
                const active = focus === id;
                return (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setFocus(id)}
                    aria-pressed={active}
                    className={cn(
                      "flex items-center gap-3 rounded-lg border p-3 text-left text-sm transition-colors",
                      active
                        ? "border-primary bg-primary/5 ring-1 ring-primary"
                        : "border-border hover:bg-muted/50",
                    )}
                  >
                    <Icon className={cn("h-5 w-5 shrink-0", active ? "text-primary" : "text-muted-foreground")} />
                    <span className="font-medium">{label}</span>
                  </button>
                );
              })}
            </CardContent>
          </Card>
        )}

        {step === 3 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-xl">Build your daily check-in</CardTitle>
              <CardDescription>
                Pick what you want to track each day. You can change this later in Settings.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {CARD_CHOICES.map((c) => {
                const active = cards[c.id];
                return (
                  <label
                    key={c.id}
                    className={cn(
                      "flex cursor-pointer items-center gap-3 rounded-lg border p-3 transition-colors",
                      active ? "border-primary/50 bg-primary/5" : "border-border hover:bg-muted/50",
                    )}
                  >
                    <Checkbox
                      checked={active}
                      onCheckedChange={(v) => setCards((prev) => ({ ...prev, [c.id]: !!v }))}
                    />
                    <div>
                      <p className="text-sm font-medium">{c.label}</p>
                      <p className="text-xs text-muted-foreground">{c.description}</p>
                    </div>
                  </label>
                );
              })}
            </CardContent>
          </Card>
        )}

        {step === 4 && (
          <Card>
            <CardHeader className="text-center">
              <div className="mb-3 flex justify-center">
                <div className="rounded-full bg-primary/10 p-3">
                  <Sparkles className="h-8 w-8 text-primary" />
                </div>
              </div>
              <CardTitle className="text-2xl">You&apos;re ready</CardTitle>
              <CardDescription className="mx-auto max-w-sm text-balance">
                Complete your daily check-in for 10 days to build your personal baseline.
                After that, MindMap can begin showing simple trends.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center gap-2 rounded-lg bg-muted/60 p-3 text-sm">
                <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" />
                <span>Day 1 of your 10-day baseline starts now.</span>
              </div>
              {error && <p className="text-center text-sm text-destructive">{error}</p>}
            </CardContent>
          </Card>
        )}
      </div>

      {/* Navigation */}
      <div className="flex items-center gap-3">
        {step > 0 && (
          <Button variant="outline" onClick={back} disabled={isPending} className="flex-1">
            Back
          </Button>
        )}
        {step < TOTAL_STEPS - 1 ? (
          <Button onClick={next} disabled={!canContinue} className="flex-1" size="lg">
            Continue
          </Button>
        ) : (
          <Button onClick={finish} disabled={isPending} className="flex-1" size="lg">
            {isPending ? <Loader2 className="animate-spin" /> : "Start Day 1 Check-In"}
          </Button>
        )}
      </div>
    </div>
  );
}
