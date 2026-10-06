import { test } from "node:test";
import assert from "node:assert/strict";
import {
  availablePresets,
  normalizeRoutineName,
  ROUTINE_PRESETS,
  ROUTINE_CATEGORY_ORDER,
} from "../lib/routine-presets";

test("with no existing routines, every preset is offered in category order", () => {
  const groups = availablePresets([]);
  assert.deepEqual(
    groups.map((g) => g.category),
    ROUTINE_CATEGORY_ORDER,
  );
  const total = groups.reduce((n, g) => n + g.presets.length, 0);
  assert.equal(total, ROUTINE_PRESETS.length);
});

test("an already-added preset is filtered out", () => {
  const groups = availablePresets(["Journaling"]);
  const names = groups.flatMap((g) => g.presets.map((p) => p.name));
  assert.ok(!names.includes("Journaling"));
  assert.ok(names.includes("Gratitude note"));
});

test("duplicate detection ignores casing and extra whitespace", () => {
  const groups = availablePresets(["  lights   OUT by 11PM "]);
  const names = groups.flatMap((g) => g.presets.map((p) => p.name));
  assert.ok(!names.includes("Lights out by 11pm"));
});

test("a category with all presets taken is omitted entirely", () => {
  const sleep = ROUTINE_PRESETS.filter((p) => p.category === "Sleep").map(
    (p) => p.name,
  );
  const groups = availablePresets(sleep);
  assert.ok(!groups.some((g) => g.category === "Sleep"));
});

test("custom routine names never collide with presets", () => {
  const groups = availablePresets(["My own thing", "Another"]);
  const total = groups.reduce((n, g) => n + g.presets.length, 0);
  assert.equal(total, ROUTINE_PRESETS.length);
});

test("normalizeRoutineName collapses whitespace and lowercases", () => {
  assert.equal(normalizeRoutineName("  Morning   Walk "), "morning walk");
});
