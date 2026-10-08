/** Calendar dates in a timezone. Entry dates are YYYY-MM-DD, not UTC instants. */

export const DEFAULT_TIMEZONE = "America/New_York";

const ISO_DATE = /^(\d{4})-(\d{2})-(\d{2})$/;

export function safeTimeZone(timeZone: string | null | undefined): string {
  const candidate = timeZone?.trim() || DEFAULT_TIMEZONE;
  try {
    Intl.DateTimeFormat("en-US", { timeZone: candidate });
    return candidate;
  } catch {
    return DEFAULT_TIMEZONE;
  }
}

/** YYYY-MM-DD for `now` as a wall-clock day in `timeZone`. */
export function calendarDate(timeZone: string | null | undefined, now = new Date()): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: safeTimeZone(timeZone),
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(now);
  const year = parts.find((part) => part.type === "year")?.value;
  const month = parts.find((part) => part.type === "month")?.value;
  const day = parts.find((part) => part.type === "day")?.value;
  if (!year || !month || !day) return now.toISOString().slice(0, 10);
  return `${year}-${month}-${day}`;
}

/** The browser's current calendar day. For client forms only. */
export function browserCalendarDate(now = new Date()): string {
  return calendarDate(Intl.DateTimeFormat().resolvedOptions().timeZone, now);
}

export function formatCalendarHeading(timeZone: string | null | undefined, now = new Date()): string {
  return new Intl.DateTimeFormat("en-US", {
    timeZone: safeTimeZone(timeZone),
    weekday: "long",
    month: "long",
    day: "numeric",
  }).format(now);
}

/** Shift a YYYY-MM-DD by whole calendar days. Does not use the machine timezone. */
export function addCalendarDays(isoDate: string, days: number): string {
  const match = ISO_DATE.exec(isoDate);
  if (!match) return isoDate;
  const utc = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]) + days));
  return utc.toISOString().slice(0, 10);
}

/**
 * Consecutive logged days ending today, or yesterday if today is still open.
 * `today` must be the viewer's calendar day, in the same zone the rows were written.
 */
export function computeStreak(dates: string[], today: string): number {
  const set = new Set(dates);
  let cursor = set.has(today) ? today : addCalendarDays(today, -1);
  let streak = 0;
  const guard = dates.length + 2;
  while (set.has(cursor) && streak < guard) {
    streak += 1;
    cursor = addCalendarDays(cursor, -1);
  }
  return streak;
}
