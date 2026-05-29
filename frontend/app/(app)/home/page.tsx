import Link from "next/link";
import { getHomeData } from "./actions";
import { getPlanProgress } from "@/lib/guided-plan";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ArrowRight,
  BarChart3,
  BookOpen,
  CheckCircle2,
  Flame,
  Lightbulb,
  ListChecks,
  Pill,
  Settings,
  Sparkles,
} from "lucide-react";

const SECONDARY_LINKS = [
  { href: "/dashboard", label: "History", icon: BarChart3 },
  { href: "/journal", label: "Journal", icon: BookOpen },
  { href: "/medications", label: "Meds", icon: Pill },
  { href: "/routines", label: "Routines", icon: ListChecks },
  { href: "/insights", label: "Reports", icon: Lightbulb },
  { href: "/settings", label: "Settings", icon: Settings },
];

export default async function HomePage() {
  const { todayScore, todayDone, checkInsCompleted, streak, latestInsight } = await getHomeData();
  const plan = getPlanProgress(checkInsCompleted);
  const phaseLength = plan.phaseRange[1] - plan.phaseRange[0] + 1;

  return (
    <div className="space-y-5">
      {/* 1. Today's check-in — the one thing to do */}
      <Card className="border-primary/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            {todayDone ? (
              <CheckCircle2 className="h-5 w-5 text-primary" />
            ) : (
              <Sparkles className="h-5 w-5 text-primary" />
            )}
            {todayDone ? "Today's check-in is done" : "Today's check-in"}
          </CardTitle>
          <CardDescription>
            {todayDone
              ? "You can update it any time before the day ends."
              : "A guided check-in takes under 90 seconds."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button asChild size="lg" className="w-full">
            <Link href="/today">
              {todayDone ? "Update check-in" : "Start check-in"}
              <ArrowRight />
            </Link>
          </Button>
        </CardContent>
      </Card>

      {/* 2 + 3. Plan progress and MindMap Score side by side */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Card>
          <CardHeader className="pb-3">
            <CardDescription>{plan.phaseLabel}</CardDescription>
            <CardTitle className="text-lg">
              Day {plan.displayDay} of {plan.phaseTotalDays}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Progress value={(plan.daysIntoPhase / phaseLength) * 100} className="h-1.5" />
            {plan.isBaselineUnlocked ? (
              <Link href="/baseline" className="inline-flex items-center gap-1 text-sm font-medium text-primary">
                View Baseline Report <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            ) : (
              <p className="text-xs text-muted-foreground">
                {plan.baselineRemaining} more {plan.baselineRemaining === 1 ? "day" : "days"} to your Baseline Report.
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardDescription>MindMap Score</CardDescription>
            <CardTitle className="text-lg">
              {todayScore != null ? `${todayScore} today` : "Not yet today"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
              <Flame className={streak > 0 ? "h-4 w-4 text-orange-500" : "h-4 w-4"} />
              {streak > 0 ? `${streak}-day streak` : "Start a streak today"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* 4. One insight card — cautious, non-diagnostic */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Lightbulb className="h-4 w-4 text-primary" /> What MindMap noticed
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {latestInsight && (latestInsight.summary || latestInsight.recommendation) ? (
            <p className="text-sm">
              {latestInsight.summary || latestInsight.recommendation}
            </p>
          ) : (
            <p className="text-sm text-muted-foreground">
              {plan.isBaselineUnlocked
                ? "Keep checking in — fresh patterns appear as you log more days."
                : "Patterns appear once you've built a 10-day baseline. Keep going!"}
            </p>
          )}
          <MedicalDisclaimer variant="compact" />
        </CardContent>
      </Card>

      {/* Secondary — accessible, not primary */}
      <div className="grid grid-cols-3 gap-2">
        {SECONDARY_LINKS.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className="flex flex-col items-center gap-1 rounded-lg border border-border p-3 text-xs text-muted-foreground transition-colors hover:bg-muted/60"
          >
            <Icon className="h-4 w-4" />
            {label}
          </Link>
        ))}
      </div>
    </div>
  );
}
