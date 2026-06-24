/**
 * App color themes ("color moods") for MindMap.
 *
 * Each theme maps to a `[data-app-theme="<id>"]` block in globals.css that
 * overrides the relevant design tokens (--primary, --accent, --ring, the
 * body gradient, etc). Components never hard-code colors — they read the
 * tokens — so switching the `data-app-theme` attribute restyles the whole app.
 *
 * The `gradient` string is a set of Tailwind classes used only for the small
 * preview swatches and accent flourishes (never behind body text).
 */

export const APP_THEME_IDS = [
  "aurora",
  "ocean",
  "lavender",
  "rose",
  "graphite",
  "forest",
] as const;

export type AppThemeId = (typeof APP_THEME_IDS)[number];

export const DEFAULT_THEME: AppThemeId = "aurora";

export type AppTheme = {
  id: AppThemeId;
  name: string;
  label: string;
  /** Tailwind gradient classes for preview swatches / accents only. */
  gradient: string;
};

export const THEMES: Record<AppThemeId, AppTheme> = {
  aurora: {
    id: "aurora",
    name: "Aurora",
    label: "Soft neon clarity",
    gradient: "from-pink-400 via-violet-400 to-sky-400",
  },
  ocean: {
    id: "ocean",
    name: "Ocean",
    label: "Calm focus",
    gradient: "from-cyan-400 via-blue-500 to-teal-400",
  },
  lavender: {
    id: "lavender",
    name: "Lavender",
    label: "Gentle reflection",
    gradient: "from-violet-300 via-purple-400 to-indigo-400",
  },
  rose: {
    id: "rose",
    name: "Rose",
    label: "Warm energy",
    gradient: "from-rose-300 via-pink-400 to-orange-300",
  },
  graphite: {
    id: "graphite",
    name: "Graphite",
    label: "Minimal focus",
    gradient: "from-slate-600 via-slate-800 to-zinc-700",
  },
  forest: {
    id: "forest",
    name: "Forest",
    label: "Grounded balance",
    gradient: "from-emerald-300 via-teal-500 to-green-700",
  },
};

export const THEME_LIST: AppTheme[] = APP_THEME_IDS.map((id) => THEMES[id]);

/** Narrow an untrusted value (e.g. from the DB) to a valid theme id. */
export function normalizeTheme(value: unknown): AppThemeId {
  return APP_THEME_IDS.includes(value as AppThemeId)
    ? (value as AppThemeId)
    : DEFAULT_THEME;
}
