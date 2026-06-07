import { test } from "node:test";
import assert from "node:assert/strict";
import {
  calculateMindMapScore,
  getMindMapScoreBreakdown,
  SCORE_WEIGHTS,
  type ScoreableEntry,
} from "../lib/mindmap-score";

test("completing a check-in always awards the base points", () => {
  assert.equal(calculateMindMapScore({}, {}), SCORE_WEIGHTS.checkInCompleted);
});

test("a fully-logged day reaches 100", () => {
  const entry: ScoreableEntry = {
    sleep_minutes: 450,
    mood_valence: 1,
    anxiety: 2,
    migraine: false,
    migraine_intensity: 0,
    notes: "felt good",
  };
  const score = calculateMindMapScore(entry, {
    medicationLogged: true,
    routineLogged: true,
    bodySymptomLogged: true,
    journalLogged: true,
  });
  assert.equal(score, 100);
});

// Non-negotiable: a bad health day must be able to score 100.
test("a migraine day can still score 100", () => {
  const migraineDay: ScoreableEntry = {
    sleep_minutes: 300,
    anxiety: 9,
    depression: 8,
    migraine: true,
    migraine_intensity: 10,
    notes: "rough day, logged it anyway",
  };
  const score = calculateMindMapScore(migraineDay, {
    medicationLogged: true,
    routineLogged: true,
    journalLogged: true,
  });
  assert.equal(score, 100);
});

// Non-negotiable: never deduct for "bad" values — only logging matters.
test("symptom severity never lowers the score vs a mild day with same logging", () => {
  const mild: ScoreableEntry = { mood_valence: 1, migraine_intensity: 1 };
  const severe: ScoreableEntry = { mood_valence: -5, migraine_intensity: 10 };
  assert.equal(
    calculateMindMapScore(severe, {}),
    calculateMindMapScore(mild, {}),
  );
});

test("each section contributes exactly its weight", () => {
  const b = getMindMapScoreBreakdown(
    { sleep_minutes: 400, mood_valence: 0, migraine_intensity: 3, notes: "x" },
    { medicationLogged: true },
  );
  assert.equal(b.checkInCompleted, SCORE_WEIGHTS.checkInCompleted);
  assert.equal(b.sleep, SCORE_WEIGHTS.sleep);
  assert.equal(b.moodFocus, SCORE_WEIGHTS.moodFocus);
  assert.equal(b.medsRoutines, SCORE_WEIGHTS.medsRoutines);
  assert.equal(b.symptom, SCORE_WEIGHTS.symptom);
  assert.equal(b.journal, SCORE_WEIGHTS.journal);
});

test("unlogged sections contribute zero", () => {
  const b = getMindMapScoreBreakdown({}, {});
  assert.equal(b.sleep, 0);
  assert.equal(b.moodFocus, 0);
  assert.equal(b.medsRoutines, 0);
  assert.equal(b.symptom, 0);
  assert.equal(b.journal, 0);
});

test("journal counts via free-text notes OR an explicit journal entry", () => {
  assert.equal(getMindMapScoreBreakdown({ notes: "  " }, {}).journal, 0); // whitespace-only doesn't count
  assert.equal(getMindMapScoreBreakdown({ notes: "real reflection" }, {}).journal, SCORE_WEIGHTS.journal);
  assert.equal(getMindMapScoreBreakdown({}, { journalLogged: true }).journal, SCORE_WEIGHTS.journal);
});

test("score is capped at 100", () => {
  const everything: ScoreableEntry = {
    sleep_minutes: 480,
    sleep_quality: 9,
    mood_valence: 2,
    anxiety: 1,
    focus: 8,
    productivity: 8,
    migraine_intensity: 2,
    notes: "all logged",
  };
  const score = calculateMindMapScore(everything, {
    medicationLogged: true,
    routineLogged: true,
    bodySymptomLogged: true,
    journalLogged: true,
  });
  assert.ok(score <= 100);
});
