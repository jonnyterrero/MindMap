/**
 * Curated starter routines offered as a quick-pick dropdown on /routines.
 * Users can still type their own; these just remove the blank-page friction.
 * Pure data + a filter helper — no DB coupling, so it's unit-testable.
 */

export type RoutineCategory =
  | "Sleep"
  | "Movement"
  | "Mind"
  | "Health"
  | "Nutrition";

export type RoutinePreset = {
  name: string;
  category: RoutineCategory;
};

/** Display order for the grouped dropdown. */
export const ROUTINE_CATEGORY_ORDER: RoutineCategory[] = [
  "Sleep",
  "Movement",
  "Mind",
  "Health",
  "Nutrition",
];

export const ROUTINE_PRESETS: RoutinePreset[] = [
  { name: "Lights out by 11pm", category: "Sleep" },
  { name: "No screens 30 min before bed", category: "Sleep" },
  { name: "Wake at a consistent time", category: "Sleep" },
  { name: "Wind-down routine", category: "Sleep" },

  { name: "20-minute walk", category: "Movement" },
  { name: "Stretch for 10 minutes", category: "Movement" },
  { name: "Workout", category: "Movement" },
  { name: "Stand up every hour", category: "Movement" },

  { name: "5-minute meditation", category: "Mind" },
  { name: "Gratitude note", category: "Mind" },
  { name: "Breathing exercise", category: "Mind" },
  { name: "Journaling", category: "Mind" },

  { name: "Take medication", category: "Health" },
  { name: "Take vitamins", category: "Health" },
  { name: "Daily check-in", category: "Health" },
  { name: "Get morning sunlight", category: "Health" },

  { name: "Drink 8 glasses of water", category: "Nutrition" },
  { name: "Eat a vegetable", category: "Nutrition" },
  { name: "No caffeine after 2pm", category: "Nutrition" },
  { name: "Eat breakfast", category: "Nutrition" },
];

/** Canonicalize a routine name for duplicate detection: trim, collapse inner
 *  whitespace, lowercase. Keeps "Morning  Walk " from re-adding "morning walk". */
export function normalizeRoutineName(name: string): string {
  return name.trim().replace(/\s+/g, " ").toLowerCase();
}

/**
 * Presets the user hasn't added yet, grouped by category in display order.
 * Groups with nothing left are omitted so the dropdown never shows an empty
 * heading. `existingNames` is the user's current routine names (any casing).
 */
export function availablePresets(
  existingNames: Iterable<string>,
): { category: RoutineCategory; presets: RoutinePreset[] }[] {
  const taken = new Set<string>();
  for (const name of existingNames) taken.add(normalizeRoutineName(name));

  return ROUTINE_CATEGORY_ORDER.map((category) => ({
    category,
    presets: ROUTINE_PRESETS.filter(
      (p) => p.category === category && !taken.has(normalizeRoutineName(p.name)),
    ),
  })).filter((group) => group.presets.length > 0);
}
