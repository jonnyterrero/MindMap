type EntryRow = Record<string, unknown>;

export type InsightResult = {
  insight_type: string;
  risk_level: string;
  score: number;
  reasons: string[];
  signals: Record<string, unknown>;
  recommendation: string | null;
};

export function computeMigraineRisk(entries: EntryRow[]): InsightResult {
  if (entries.length === 0) {
    return {
      insight_type: "migraine_risk",
      risk_level: "unknown",
      score: 0,
      reasons: ["Not enough data"],
      signals: {},
      recommendation: "Log a few days of data to see predictions.",
    };
  }

  const reasons: string[] = [];
  const signals: Record<string, unknown> = {};
  let score = 0;

  const latest = entries[0];
  const sleepMinutes = (latest.sleep_minutes as number) ?? 480;
  const sleepHrs = sleepMinutes / 60;
  const anxiety = (latest.anxiety as number) ?? 0;
  const depression = (latest.depression as number) ?? 0;

  if (sleepHrs < 6) {
    score += 30;
    reasons.push(`Sleep was only ${sleepHrs.toFixed(1)}h (< 6h)`);
    signals.sleep_hours = sleepHrs;
  } else if (sleepHrs < 7) {
    score += 15;
    reasons.push(`Sleep was ${sleepHrs.toFixed(1)}h (< 7h)`);
    signals.sleep_hours = sleepHrs;
  }

  if (anxiety >= 7) {
    score += 25;
    reasons.push(`Anxiety is high (${anxiety}/10)`);
    signals.anxiety = anxiety;
  } else if (anxiety >= 5) {
    score += 10;
    reasons.push(`Anxiety is moderate (${anxiety}/10)`);
    signals.anxiety = anxiety;
  }

  if (depression >= 7) {
    score += 15;
    reasons.push(`Depression is elevated (${depression}/10)`);
    signals.depression = depression;
  }

  const recentMigraines = entries
    .slice(0, 7)
    .filter((e) => e.migraine === true).length;
  if (recentMigraines >= 3) {
    score += 30;
    reasons.push(`${recentMigraines} migraines in the last 7 days`);
    signals.recent_migraines = recentMigraines;
  } else if (recentMigraines >= 1) {
    score += 10;
    reasons.push(`${recentMigraines} migraine(s) in the last 7 days`);
    signals.recent_migraines = recentMigraines;
  }

  const sleepVariance = computeSleepVariance(entries.slice(0, 7));
  if (sleepVariance > 90) {
    score += 15;
    reasons.push("Sleep schedule is irregular (high variance)");
    signals.sleep_variance_minutes = Math.round(sleepVariance);
  }

  score = Math.min(score, 100);

  let risk_level: string;
  let recommendation: string;
  if (score >= 60) {
    risk_level = "high";
    recommendation = "Consider rest, hydration, and avoiding triggers today.";
  } else if (score >= 30) {
    risk_level = "moderate";
    recommendation = "Monitor for early migraine signs. Prioritize good sleep tonight.";
  } else {
    risk_level = "low";
    recommendation = "Looking good! Keep up your current routine.";
  }

  if (reasons.length === 0) {
    reasons.push("No risk factors detected");
  }

  return { insight_type: "migraine_risk", risk_level, score, reasons, signals, recommendation };
}

export function computeMoodTrend(entries: EntryRow[]): InsightResult {
  if (entries.length < 3) {
    return {
      insight_type: "mood_trend",
      risk_level: "unknown",
      score: 0,
      reasons: ["Need at least 3 days of data"],
      signals: {},
      recommendation: "Keep logging daily.",
    };
  }

  const recent = entries.slice(0, 7);
  const avgAnxiety = avg(recent.map((e) => (e.anxiety as number) ?? 0));
  const avgDepression = avg(recent.map((e) => (e.depression as number) ?? 0));
  const avgFocus = avg(recent.map((e) => (e.focus as number) ?? 5));
  const avgProductivity = avg(recent.map((e) => (e.productivity as number) ?? 5));

  const reasons: string[] = [];
  const signals: Record<string, unknown> = { avgAnxiety, avgDepression, avgFocus, avgProductivity };
  let score = 0;

  if (avgAnxiety >= 6) { score += 30; reasons.push(`Average anxiety is ${avgAnxiety.toFixed(1)}/10`); }
  if (avgDepression >= 6) { score += 30; reasons.push(`Average depression is ${avgDepression.toFixed(1)}/10`); }
  if (avgFocus <= 3) { score += 15; reasons.push(`Average focus is low (${avgFocus.toFixed(1)}/10)`); }
  if (avgProductivity <= 3) { score += 15; reasons.push(`Average productivity is low (${avgProductivity.toFixed(1)}/10)`); }

  score = Math.min(score, 100);

  let risk_level: string;
  let recommendation: string;
  if (score >= 50) {
    risk_level = "concerning";
    recommendation = "Consider discussing recent mood patterns with a therapist.";
  } else if (score >= 20) {
    risk_level = "moderate";
    recommendation = "Some mood fluctuation detected. Self-care practices may help.";
  } else {
    risk_level = "stable";
    recommendation = "Your mood has been steady. Great job maintaining your routines!";
  }

  if (reasons.length === 0) reasons.push("Mood metrics are within healthy ranges");

  return { insight_type: "mood_trend", risk_level, score, reasons, signals, recommendation };
}

function avg(nums: number[]): number {
  return nums.length === 0 ? 0 : nums.reduce((a, b) => a + b, 0) / nums.length;
}

function computeSleepVariance(entries: EntryRow[]): number {
  const sleeps = entries
    .map((e) => (e.sleep_minutes as number) ?? null)
    .filter((s): s is number => s !== null);
  if (sleeps.length < 2) return 0;
  const mean = avg(sleeps);
  const variance = sleeps.reduce((sum, v) => sum + (v - mean) ** 2, 0) / sleeps.length;
  return Math.sqrt(variance);
}
