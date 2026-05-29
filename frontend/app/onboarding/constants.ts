// Plain (non-"use server") module: shared constants/types for onboarding.
// A "use server" file may only export async functions, so these live here.

export const FOCUS_OPTIONS = [
  "migraine",
  "anxiety",
  "adhd",
  "mood",
  "sleep",
  "medication",
] as const;

export const CHECKIN_CARDS = [
  "sleep",
  "mood",
  "focus",
  "migraine",
  "medication",
  "routines",
  "journal",
] as const;

export type FocusOption = (typeof FOCUS_OPTIONS)[number];
export type CheckinCard = (typeof CHECKIN_CARDS)[number];
