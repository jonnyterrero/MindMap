import { test } from "node:test";
import assert from "node:assert/strict";
import { addCalendarDays, calendarDate, computeStreak } from "../lib/local-date";

// 2026-10-08 01:00 UTC is still 2026-10-07 in New York (EDT, UTC-4).
const EVENING_NY = new Date("2026-10-08T01:00:00.000Z");

test("calendarDate uses the profile timezone, not UTC", () => {
  assert.equal(calendarDate("America/New_York", EVENING_NY), "2026-10-07");
  assert.equal(calendarDate("UTC", EVENING_NY), "2026-10-08");
});

test("calendarDate falls back when the timezone is invalid", () => {
  assert.equal(calendarDate("Not/AZone", EVENING_NY), "2026-10-07");
  assert.equal(calendarDate(null, EVENING_NY), "2026-10-07");
});

test("addCalendarDays walks the calendar across month ends", () => {
  assert.equal(addCalendarDays("2026-10-07", -1), "2026-10-06");
  assert.equal(addCalendarDays("2026-03-01", -1), "2026-02-28");
  assert.equal(addCalendarDays("2024-03-01", -1), "2024-02-29");
});

test("computeStreak counts through today and gives yesterday a one-day grace", () => {
  assert.equal(computeStreak(["2026-10-07", "2026-10-06", "2026-10-05"], "2026-10-07"), 3);
  assert.equal(computeStreak(["2026-10-06", "2026-10-05"], "2026-10-07"), 2);
  assert.equal(computeStreak(["2026-10-05"], "2026-10-07"), 0);
  assert.equal(computeStreak([], "2026-10-07"), 0);
});
