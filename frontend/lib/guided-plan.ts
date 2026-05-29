/**
 * Guided Plan
 * -----------
 * MindMap onboards users with a 30-day guided plan built around tracking
 * consistency. Progress is measured by *check-in days completed*, not the
 * calendar — missing a day never penalizes the user, it just means the plan
 * takes longer to advance.
 *
 *   Days  1–10  Baseline Tracking      (unlocks the Day-10 Baseline Report)
 *   Days 11–20  Pattern Discovery
 *   Days 21–30  Personal Optimization
 */

export type PlanPhase = "baseline" | "pattern_discovery" | "optimization";

export interface PlanProgress {
  /** Distinct check-in days completed so far. */
  checkInsCompleted: number;
  /** 1-based day to display ("Day X of N"). */
  displayDay: number;
  phase: PlanPhase;
  phaseLabel: string;
  /** Inclusive [start, end] day numbers for the current phase. */
  phaseRange: [number, number];
  /** "Day X of 10" — the end of the current phase, for the progress label. */
  phaseTotalDays: number;
  /** Days completed within the current phase. */
  daysIntoPhase: number;
  /** True once 10 check-ins exist (Baseline Report available). */
  isBaselineUnlocked: boolean;
  /** Check-ins remaining until the baseline report unlocks (0 once unlocked). */
  baselineRemaining: number;
  /** Total length of the guided plan. */
  totalPlanDays: number;
}

export const PLAN_TOTAL_DAYS = 30;
export const BASELINE_DAYS = 10;

const PHASES: Record<PlanPhase, { label: string; range: [number, number] }> = {
  baseline: { label: "Baseline Tracking", range: [1, 10] },
  pattern_discovery: { label: "Pattern Discovery", range: [11, 20] },
  optimization: { label: "Personal Optimization", range: [21, 30] },
};

function phaseFor(day: number): PlanPhase {
  if (day <= 10) return "baseline";
  if (day <= 20) return "pattern_discovery";
  return "optimization";
}

/**
 * Compute guided-plan progress from the number of check-in days completed.
 */
export function getPlanProgress(checkInsCompleted: number): PlanProgress {
  const completed = Math.max(0, Math.floor(checkInsCompleted));

  // Display day: before the first check-in we show "Day 1" (upcoming);
  // afterward it tracks check-ins, capped at the plan length.
  const displayDay = Math.min(Math.max(1, completed), PLAN_TOTAL_DAYS);

  const phase = phaseFor(displayDay);
  const { label, range } = PHASES[phase];
  const daysIntoPhase = Math.max(0, completed - (range[0] - 1));

  return {
    checkInsCompleted: completed,
    displayDay,
    phase,
    phaseLabel: label,
    phaseRange: range,
    phaseTotalDays: range[1],
    daysIntoPhase,
    isBaselineUnlocked: completed >= BASELINE_DAYS,
    baselineRemaining: Math.max(0, BASELINE_DAYS - completed),
    totalPlanDays: PLAN_TOTAL_DAYS,
  };
}
