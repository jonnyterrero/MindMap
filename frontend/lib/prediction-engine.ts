/**
 * Predictive Engine v1 (v1_rule_extended)
 * ---------------------------------------
 * Wraps the existing rule-based `insights-engine` heuristics and layers on
 * wearable + weather + recurrence signals to produce 0..1 risk scores for
 * migraine / anxiety / mood / pain_flare. Pure & deterministic — all DB I/O
 * and persistence live in the server action (`prediction-actions.ts`).
 *
 * Wellness self-tracking only — not a diagnosis (see MedicalDisclaimer).
 */
import { computeMigraineRisk, computeMoodTrend } from "./insights-engine";

export const MODEL_VERSION = "v1_rule_extended";

export type PredictionType = "migraine" | "anxiety" | "mood" | "pain_flare";
export type RiskLevel = "low" | "moderate" | "high" | "critical";

export interface ContributingFactor {
  factor: string;
  weight: number; // signed contribution to the 0..1 score
  detail?: string;
}

export interface ComputedPrediction {
  prediction_type: PredictionType;
  risk_score: number; // 0..1
  risk_level: RiskLevel;
  confidence: number; // 0..1
  contributing_factors: ContributingFactor[];
  model_version: string;
}

export interface WearableLatest {
  hrv?: number | null;
  sleep_score?: number | null;
  resting_hr?: number | null;
}

export interface WeatherSignal {
  pressure_change?: number | null; // hPa over 24h; negative = drop
  pollen_level?: string | null; // low|moderate|high|very_high
}

export interface PredictionInput {
  entries: Record<string, unknown>[]; // newest first, up to ~14 days
  wearable?: WearableLatest;
  weather?: WeatherSignal;
}

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));
const num = (v: unknown): number | null => (typeof v === "number" ? v : null);

function levelFor(score: number): RiskLevel {
  if (score > 0.8) return "critical";
  if (score > 0.6) return "high";
  if (score > 0.3) return "moderate";
  return "low";
}

function avg(nums: (number | null)[]): number {
  const v = nums.filter((n): n is number => n !== null);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0;
}

/** Recent-window helpers (entries are newest-first). */
function recurrenceCount(entries: Record<string, unknown>[], pred: (e: Record<string, unknown>) => boolean): number {
  return entries.slice(0, 7).filter(pred).length;
}

function applyWearableAndWeather(
  type: PredictionType,
  base: number,
  factors: ContributingFactor[],
  wearable: WearableLatest | undefined,
  weather: WeatherSignal | undefined,
): number {
  let score = base;

  const hrv = wearable?.hrv ?? null;
  const sleepScore = wearable?.sleep_score ?? null;
  const restingHr = wearable?.resting_hr ?? null;

  // Low HRV → nudges anxiety/migraine up.
  if (hrv != null && hrv < 40 && (type === "anxiety" || type === "migraine")) {
    score += 0.08;
    factors.push({ factor: "low_hrv", weight: 0.08, detail: `HRV ${hrv}ms (low)` });
  }
  // Poor sleep score → broad risk bump.
  if (sleepScore != null && sleepScore < 60 && type !== "pain_flare") {
    score += 0.15;
    factors.push({ factor: "low_sleep_score", weight: 0.15, detail: `Sleep score ${sleepScore} (<60)` });
  }
  // Elevated resting HR → pain flare.
  if (restingHr != null && restingHr > 85 && type === "pain_flare") {
    score += 0.1;
    factors.push({ factor: "elevated_resting_hr", weight: 0.1, detail: `Resting HR ${restingHr}bpm (>85)` });
  }
  if (hrv != null && hrv < 40 && type === "pain_flare") {
    score += 0.05;
    factors.push({ factor: "low_hrv", weight: 0.05, detail: `HRV ${hrv}ms (low)` });
  }

  // Weather: barometric pressure drop → migraine.
  if (type === "migraine") {
    const drop = weather?.pressure_change ?? null;
    if (drop != null && drop < -8) {
      score += 0.2;
      factors.push({ factor: "pressure_drop", weight: 0.2, detail: `Pressure ${drop.toFixed(1)} hPa/24h (drop)` });
    }
    const pollen = weather?.pollen_level ?? null;
    if (pollen === "high" || pollen === "very_high") {
      score += 0.15;
      factors.push({ factor: "high_pollen", weight: 0.15, detail: `Pollen ${pollen}` });
    }
  }

  return score;
}

function baseFor(type: PredictionType, entries: Record<string, unknown>[], factors: ContributingFactor[]): number {
  const recent = entries.slice(0, 7);
  switch (type) {
    case "migraine": {
      const r = computeMigraineRisk(entries);
      r.reasons.forEach((reason) => factors.push({ factor: "rule_base", weight: 0, detail: reason }));
      return r.score / 100;
    }
    case "mood": {
      const r = computeMoodTrend(entries);
      r.reasons.forEach((reason) => factors.push({ factor: "rule_base", weight: 0, detail: reason }));
      return r.score / 100;
    }
    case "anxiety": {
      const a = avg(recent.map((e) => num(e.anxiety)));
      if (a >= 5) factors.push({ factor: "recent_anxiety", weight: a / 10, detail: `Avg anxiety ${a.toFixed(1)}/10` });
      return clamp01(a / 10);
    }
    case "pain_flare": {
      const mi = avg(recent.map((e) => num(e.migraine_intensity)));
      const ax = avg(recent.map((e) => num(e.anxiety)));
      const base = clamp01((mi / 10) * 0.6 + (ax / 10) * 0.25);
      if (mi >= 3) factors.push({ factor: "recent_pain", weight: (mi / 10) * 0.6, detail: `Avg migraine intensity ${mi.toFixed(1)}/10` });
      return base;
    }
  }
}

function recurrenceFor(type: PredictionType, entries: Record<string, unknown>[]): number {
  switch (type) {
    case "migraine":
      return recurrenceCount(entries, (e) => e.migraine === true);
    case "anxiety":
      return recurrenceCount(entries, (e) => (num(e.anxiety) ?? 0) >= 7);
    case "mood":
      return recurrenceCount(entries, (e) => (num(e.mood_valence) ?? 0) < 0 || (num(e.depression) ?? 0) >= 6);
    case "pain_flare":
      return recurrenceCount(entries, (e) => (num(e.migraine_intensity) ?? 0) >= 6);
  }
}

function computeOne(type: PredictionType, input: PredictionInput): ComputedPrediction {
  const factors: ContributingFactor[] = [];
  let score = baseFor(type, input.entries, factors);
  score = applyWearableAndWeather(type, score, factors, input.wearable, input.weather);
  score = clamp01(score);

  // Confidence: data volume + recurrence + wearable presence.
  const dataDays = Math.min(input.entries.length, 14);
  let confidence = 0.4 + (dataDays / 14) * 0.3;
  const recur = recurrenceFor(type, input.entries);
  if (recur >= 3) {
    confidence += 0.12;
    factors.push({ factor: "recurrence", weight: 0, detail: `Pattern recurred ${recur}× in 7 days` });
  }
  if (input.wearable && (input.wearable.hrv != null || input.wearable.sleep_score != null)) confidence += 0.1;
  confidence = clamp01(confidence);

  return {
    prediction_type: type,
    risk_score: Math.round(score * 1000) / 1000,
    risk_level: levelFor(score),
    confidence: Math.round(confidence * 1000) / 1000,
    contributing_factors: factors,
    model_version: MODEL_VERSION,
  };
}

/** Compute predictions for all four types from the gathered input. */
export function computePredictions(input: PredictionInput): ComputedPrediction[] {
  if (input.entries.length === 0) return [];
  const types: PredictionType[] = ["migraine", "anxiety", "mood", "pain_flare"];
  return types.map((t) => computeOne(t, input));
}
