import assert from "node:assert/strict";
import test from "node:test";
import { resolveAnalyticsOptIn } from "../lib/analytics-consent";

test("analytics is off when no preference or consent exists", () => {
  assert.equal(resolveAnalyticsOptIn(null, null), false);
});

test("onboarding consent is used before a settings row exists", () => {
  assert.equal(resolveAnalyticsOptIn(null, { consent_given: true }), true);
});

test("the mutable privacy setting overrides historical consent", () => {
  assert.equal(
    resolveAnalyticsOptIn(
      { analytics_opt_in: false },
      { consent_given: true },
    ),
    false,
  );
});
