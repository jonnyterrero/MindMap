/**
 * MindMap Score
 * -------------
 * A 0–100 score that rewards tracking *consistency and self-awareness only*.
 *
 * Design rules (non-negotiable):
 *  - Never deduct points for "bad" health days (a migraine day can score 100).
 *  - Never frame a low score as a health failure. It measures logging, not health.
 *  - Showing up (completing a check-in) is always worth points.
 *
 * The score is intentionally decoupled from the raw DB row: callers pass the
 * entry fields they have plus a small context object describing signals that
 * live in related tables (routines, meds, body sensations, journal).
 */

export interface ScoreableEntry {
  sleep_minutes?: number | null;
  sleep_quality?: number | null;
  bed_time?: string | null;
  wake_time?: string | null;
  mood_valence?: number | null;
  anxiety?: number | null;
  depression?: number | null;
  mania?: number | null;
  focus?: number | null;
  productivity?: number | null;
  migraine?: boolean | null;
  migraine_intensity?: number | null;
  notes?: string | null;
}

export interface ScoreContext {
  /** Any medication marked taken or skipped today. */
  medicationLogged?: boolean;
  /** Any routine toggled today. */
  routineLogged?: boolean;
  /** Any body sensation recorded today. */
  bodySymptomLogged?: boolean;
  /** A journal / reflection entry was saved today. */
  journalLogged?: boolean;
}

export interface ScoreBreakdown {
  checkInCompleted: number;
  sleep: number;
  moodFocus: number;
  medsRoutines: number;
  symptom: number;
  journal: number;
  total: number;
}

export const SCORE_WEIGHTS = {
  checkInCompleted: 25,
  sleep: 15,
  moodFocus: 15,
  medsRoutines: 20,
  symptom: 15,
  journal: 10, // optional bonus
} as const;

function hasNumber(value: number | null | undefined): boolean {
  return value !== null && value !== undefined && !Number.isNaN(value);
}

function isSleepLogged(e: ScoreableEntry): boolean {
  return (
    hasNumber(e.sleep_minutes) ||
    hasNumber(e.sleep_quality) ||
    Boolean(e.bed_time) ||
    Boolean(e.wake_time)
  );
}

function isMoodFocusLogged(e: ScoreableEntry): boolean {
  return (
    hasNumber(e.mood_valence) ||
    hasNumber(e.anxiety) ||
    hasNumber(e.depression) ||
    hasNumber(e.mania) ||
    hasNumber(e.focus) ||
    hasNumber(e.productivity)
  );
}

function isSymptomLogged(e: ScoreableEntry, ctx: ScoreContext): boolean {
  return e.migraine === true || hasNumber(e.migraine_intensity) || Boolean(ctx.bodySymptomLogged);
}

function isJournalLogged(e: ScoreableEntry, ctx: ScoreContext): boolean {
  return Boolean(ctx.journalLogged) || Boolean(e.notes && e.notes.trim().length > 0);
}

/**
 * Returns the per-section point breakdown for a completed check-in.
 * `checkInCompleted` is always awarded — the act of checking in counts.
 */
export function getMindMapScoreBreakdown(
  entry: ScoreableEntry,
  context: ScoreContext = {},
): ScoreBreakdown {
  const sleep = isSleepLogged(entry) ? SCORE_WEIGHTS.sleep : 0;
  const moodFocus = isMoodFocusLogged(entry) ? SCORE_WEIGHTS.moodFocus : 0;
  const medsRoutines =
    context.medicationLogged || context.routineLogged ? SCORE_WEIGHTS.medsRoutines : 0;
  const symptom = isSymptomLogged(entry, context) ? SCORE_WEIGHTS.symptom : 0;
  const journal = isJournalLogged(entry, context) ? SCORE_WEIGHTS.journal : 0;

  const total = Math.min(
    100,
    SCORE_WEIGHTS.checkInCompleted + sleep + moodFocus + medsRoutines + symptom + journal,
  );

  return {
    checkInCompleted: SCORE_WEIGHTS.checkInCompleted,
    sleep,
    moodFocus,
    medsRoutines,
    symptom,
    journal,
    total,
  };
}

/**
 * Convenience wrapper returning just the 0–100 MindMap Score.
 */
export function calculateMindMapScore(
  entry: ScoreableEntry,
  context: ScoreContext = {},
): number {
  return getMindMapScoreBreakdown(entry, context).total;
}
