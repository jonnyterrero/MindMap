import { test } from "node:test";
import assert from "node:assert/strict";
import {
  computeMigraineRisk,
  computeMoodTrend,
} from "../lib/insights-engine";

test("migraine risk: empty data is 'unknown'", () => {
  const r = computeMigraineRisk([]);
  assert.equal(r.risk_level, "unknown");
  assert.equal(r.score, 0);
});

test("migraine risk: short sleep + high anxiety raises score", () => {
  const r = computeMigraineRisk([
    { sleep_minutes: 300 /* 5h */, anxiety: 8, depression: 2, migraine: false },
  ]);
  assert.ok(r.score >= 30, `expected elevated score, got ${r.score}`);
  assert.ok(["moderate", "high"].includes(r.risk_level));
  assert.ok(r.reasons.length > 0);
});

test("migraine risk: healthy day is low", () => {
  const r = computeMigraineRisk([
    { sleep_minutes: 480 /* 8h */, anxiety: 1, depression: 1, migraine: false },
  ]);
  assert.equal(r.risk_level, "low");
});

test("migraine risk: recurring migraines push score up", () => {
  const week = Array.from({ length: 7 }, () => ({
    sleep_minutes: 480,
    anxiety: 2,
    depression: 1,
    migraine: true,
  }));
  const r = computeMigraineRisk(week);
  assert.ok(r.signals.recent_migraines === 7 || r.score >= 30);
});

test("migraine risk: score is capped at 100", () => {
  const r = computeMigraineRisk([
    { sleep_minutes: 120, anxiety: 10, depression: 10, migraine: true },
  ]);
  assert.ok(r.score <= 100);
});

test("mood trend: needs >= 3 days", () => {
  const r = computeMoodTrend([{ anxiety: 5 }, { anxiety: 6 }]);
  assert.equal(r.risk_level, "unknown");
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
