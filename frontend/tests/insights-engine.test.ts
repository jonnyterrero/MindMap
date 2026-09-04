import { test } from "node:test";
import assert from "node:assert/strict";
import {
  computeMigraineRisk,
  computeMoodTrend,
  INSIGHT_MIN_ENTRIES,
} from "../lib/insights-engine";

/** A healthy filler day, overridable per test. Engines read entries[0] as latest. */
function day(overrides: Record<string, unknown> = {}) {
  return { sleep_minutes: 480, anxiety: 1, depression: 1, migraine: false, ...overrides };
}

function pad(latest: Record<string, unknown>, total = INSIGHT_MIN_ENTRIES.migraine_risk) {
  return [latest, ...Array.from({ length: total - 1 }, () => day())];
}

test("migraine risk: empty data is 'unknown'", () => {
  const r = computeMigraineRisk([]);
  assert.equal(r.risk_level, "unknown");
  assert.equal(r.score, 0);
});

test("migraine risk: below min-entries guardrail stays 'unknown' with progress copy", () => {
  const r = computeMigraineRisk([day(), day(), day()]);
  assert.equal(r.risk_level, "unknown");
  assert.equal(r.score, 0);
  assert.match(r.reasons[0], /3 of 5/);
  assert.match(r.recommendation ?? "", /2 more days/);
});

test("migraine risk: short sleep + high anxiety raises score", () => {
  const r = computeMigraineRisk(
    pad(day({ sleep_minutes: 300 /* 5h */, anxiety: 8, depression: 2 })),
  );
  assert.ok(r.score >= 30, `expected elevated score, got ${r.score}`);
  assert.ok(["moderate", "high"].includes(r.risk_level));
  assert.ok(r.reasons.length > 0);
});

test("migraine risk: healthy days are low", () => {
  const r = computeMigraineRisk(pad(day()));
  assert.equal(r.risk_level, "low");
});

test("migraine risk: recurring migraines push score up", () => {
  const week = Array.from({ length: 7 }, () => day({ migraine: true }));
  const r = computeMigraineRisk(week);
  assert.ok(r.signals.recent_migraines === 7 || r.score >= 30);
});

test("migraine risk: score is capped at 100", () => {
  const r = computeMigraineRisk(
    pad(day({ sleep_minutes: 120, anxiety: 10, depression: 10, migraine: true })),
  );
  assert.ok(r.score <= 100);
});

test("mood trend: below min-entries guardrail stays 'unknown'", () => {
  const r = computeMoodTrend([{ anxiety: 5 }, { anxiety: 6 }]);
  assert.equal(r.risk_level, "unknown");
  assert.match(r.reasons[0], /2 of 5/);
});

test("mood trend: sustained high anxiety+depression is 'concerning'", () => {
  const days = Array.from({ length: 5 }, () => ({
    anxiety: 8,
    depression: 8,
    focus: 4,
    productivity: 4,
  }));
  const r = computeMoodTrend(days);
  assert.equal(r.risk_level, "concerning");
});

test("mood trend: steady healthy mood is 'stable'", () => {
  const days = Array.from({ length: 5 }, () => ({
    anxiety: 1,
    depression: 1,
    focus: 8,
    productivity: 8,
  }));
  const r = computeMoodTrend(days);
  assert.equal(r.risk_level, "stable");
});
