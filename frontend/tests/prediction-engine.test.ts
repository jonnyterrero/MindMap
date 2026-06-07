import { test } from "node:test";
import assert from "node:assert/strict";
import {
  computePredictions,
  MODEL_VERSION,
  type PredictionInput,
} from "../lib/prediction-engine";

function baseEntries(n: number): Record<string, unknown>[] {
  return Array.from({ length: n }, () => ({
    sleep_minutes: 450,
    anxiety: 3,
    depression: 2,
    mood_valence: 0,
    focus: 6,
    productivity: 6,
    migraine: false,
    migraine_intensity: 2,
  }));
}

function painFlare(input: PredictionInput) {
  const p = computePredictions(input).find((x) => x.prediction_type === "pain_flare");
  assert.ok(p, "expected a pain_flare prediction");
  return p;
}

test("empty entries yields no predictions", () => {
  assert.deepEqual(computePredictions({ entries: [] }), []);
});

test("produces all four prediction types", () => {
  const preds = computePredictions({ entries: baseEntries(10) });
  const types = preds.map((p) => p.prediction_type).sort();
  assert.deepEqual(types, ["anxiety", "migraine", "mood", "pain_flare"]);
});

test("every prediction carries the model version and a 0..1 score", () => {
  for (const p of computePredictions({ entries: baseEntries(10) })) {
    assert.equal(p.model_version, MODEL_VERSION);
    assert.ok(p.risk_score >= 0 && p.risk_score <= 1);
    assert.ok(p.confidence >= 0 && p.confidence <= 1);
  }
});

test("logged body pain raises the pain_flare score", () => {
  const entries = baseEntries(10);
  const without = painFlare({ entries });
  const withPain = painFlare({ entries, bodyPain: { avgIntensity: 8 } });
  assert.ok(without && withPain);
  assert.ok(
    withPain.risk_score > without.risk_score,
    `expected ${withPain.risk_score} > ${without.risk_score}`,
  );
  assert.ok(
    withPain.contributing_factors.some((f) => f.factor === "logged_body_pain"),
  );
});

test("body pain at zero intensity does not change the score", () => {
  const entries = baseEntries(10);
  const without = painFlare({ entries });
  const zero = painFlare({ entries, bodyPain: { avgIntensity: 0 } });
  assert.equal(zero.risk_score, without.risk_score);
});

test("days-with-pain recurrence raises pain_flare confidence", () => {
  const entries = baseEntries(10);
  const low = painFlare({ entries, bodyPain: { avgIntensity: 5, daysWithPain: 1 } });
  const high = painFlare({ entries, bodyPain: { avgIntensity: 5, daysWithPain: 4 } });
  assert.ok(high.confidence > low.confidence);
});

test("risk score never exceeds 1 even with maximal signals", () => {
  const entries = Array.from({ length: 14 }, () => ({
    sleep_minutes: 120,
    anxiety: 10,
    depression: 10,
    migraine: true,
    migraine_intensity: 10,
  }));
  const p = painFlare({
    entries,
    wearable: { hrv: 20, resting_hr: 100 },
    bodyPain: { avgIntensity: 10, daysWithPain: 7 },
  });
  assert.ok(p.risk_score <= 1);
});
