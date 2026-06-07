import { test } from "node:test";
import assert from "node:assert/strict";
import {
  computeCorrelations,
  CORRELATION_METRICS,
} from "../lib/correlation-engine";

// Build N entries where two metrics move together perfectly.
function entries(n: number, fn: (i: number) => Record<string, unknown>) {
  return Array.from({ length: n }, (_, i) => fn(i));
}

test("body_pain is a registered correlation metric", () => {
  assert.ok(CORRELATION_METRICS.some((m) => m.key === "body_pain"));
});

test("perfect positive correlation yields r ~ 1 and 'strong'", () => {
  const data = entries(12, (i) => ({ anxiety: i, migraine_intensity: i }));
  const res = computeCorrelations(data, { minSampleSize: 8, minAbsR: 0.3 });
  const pair = res.find(
    (c) =>
      (c.aKey === "anxiety" && c.bKey === "migraine_intensity") ||
      (c.aKey === "migraine_intensity" && c.bKey === "anxiety"),
  );
  assert.ok(pair, "expected anxiety/migraine_intensity pair");
  assert.equal(pair.r, 1);
  assert.equal(pair.strength, "strong");
  assert.equal(pair.direction, "positive");
});

test("perfect negative correlation is detected", () => {
  const data = entries(12, (i) => ({ focus: i, depression: 11 - i }));
  const res = computeCorrelations(data, { minSampleSize: 8, minAbsR: 0.3 });
  const pair = res.find(
    (c) =>
      (c.aKey === "focus" && c.bKey === "depression") ||
      (c.aKey === "depression" && c.bKey === "focus"),
  );
  assert.ok(pair);
  assert.equal(pair.direction, "negative");
  assert.ok(pair.r < 0);
});

test("body_pain correlates with migraine_intensity when both present", () => {
  const data = entries(12, (i) => ({ body_pain: i, migraine_intensity: i }));
  const res = computeCorrelations(data, { minSampleSize: 8, minAbsR: 0.3 });
  const pair = res.find((c) => c.aKey === "body_pain" || c.bKey === "body_pain");
  assert.ok(pair, "expected a body_pain correlation");
  assert.equal(Math.abs(pair.r), 1);
});

test("below minimum sample size yields no results", () => {
  const data = entries(5, (i) => ({ anxiety: i, depression: i }));
  const res = computeCorrelations(data, { minSampleSize: 8 });
  assert.equal(res.length, 0);
});

test("zero-variance series produce no correlation", () => {
  const data = entries(12, () => ({ anxiety: 5, depression: 5 }));
  const res = computeCorrelations(data);
  assert.equal(res.length, 0);
});
