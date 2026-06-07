import { test } from "node:test";
import assert from "node:assert/strict";
import {
  detectCrisis,
  crisisHeader,
  CRISIS_RESOURCES,
} from "../lib/crisis-detection";

test("detects critical intent", () => {
  assert.equal(detectCrisis("I want to kill myself"), "critical");
  assert.equal(detectCrisis("thinking about suicide"), "critical");
  assert.equal(detectCrisis("I'd be better off dead"), "critical");
});

test("detects moderate (self-harm) intent", () => {
  assert.equal(detectCrisis("I keep wanting to hurt myself"), "moderate");
  assert.equal(detectCrisis("thoughts of self-harm"), "moderate");
});

test("detects concern-level language", () => {
  assert.equal(detectCrisis("I feel hopeless today"), "concern");
  assert.equal(detectCrisis("totally overwhelmed and can't cope"), "concern");
});

test("returns null for benign text", () => {
  assert.equal(detectCrisis("had a great walk and good sleep"), null);
  assert.equal(detectCrisis("looking forward to the weekend"), null);
});

test("substring matching is intentional (errs toward showing help)", () => {
  // "hopelessly" contains the "hopeless" phrase — conservative by design.
  assert.equal(detectCrisis("hopelessly romantic evening"), "concern");
});

test("handles empty / null / undefined", () => {
  assert.equal(detectCrisis(""), null);
  assert.equal(detectCrisis(null), null);
  assert.equal(detectCrisis(undefined), null);
});

test("highest tier wins when multiple tiers present", () => {
  // contains both a concern word ("hopeless") and a critical phrase
  assert.equal(detectCrisis("I feel hopeless and want to die"), "critical");
  // concern + moderate -> moderate
  assert.equal(detectCrisis("overwhelmed and want to hurt myself"), "moderate");
});

test("is case-insensitive and whitespace-normalized", () => {
  assert.equal(detectCrisis("KILL   MYSELF"), "critical");
  assert.equal(detectCrisis("Want\nTo\tDie"), "critical");
});

test("crisisHeader returns title + body for each severity", () => {
  for (const sev of ["concern", "moderate", "critical"] as const) {
    const h = crisisHeader(sev);
    assert.ok(h.title.length > 0);
    assert.ok(h.body.length > 0);
  }
});

test("crisis resources include the 988 lifeline", () => {
  assert.ok(CRISIS_RESOURCES.some((r) => r.detail.includes("988")));
});
