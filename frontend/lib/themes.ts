/**
 * App color themes ("color moods") for MindMap.
 *
 * Each theme maps to a `[data-app-theme="<id>"]` block in globals.css that
 * overrides the relevant design tokens (--primary, --accent, --ring, the
 * body gradient, etc). Components never hard-code colors — they read the
 * tokens — so switching the `data-app-theme` attribute restyles the whole app.
 *
 * The attribute lives on <html>, not on a wrapper element: Radix renders
 * dialogs/dropdowns/selects/popovers into a portal on <body>, and <body>
 * itself paints the themed gradient — both sit outside any wrapper div.
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

/**
 * Mirrors the profile's `app_theme` so the very first paint of any page can
 * pick up the right palette without a DB round-trip. The profile row stays the
 * source of truth across devices; this is only a rendering hint.
 */
export const THEME_COOKIE = "app_theme";
export const THEME_COOKIE_MAX_AGE = 60 * 60 * 24 * 365; // 1 year

/** Narrow an untrusted value (e.g. from the DB or a cookie) to a valid id. */
export function normalizeTheme(value: unknown): AppThemeId {
  return APP_THEME_IDS.includes(value as AppThemeId)
    ? (value as AppThemeId)
    : DEFAULT_THEME;
}

/**
 * Runs blocking in <head> before first paint so the themed palette is already
 * on <html> when the page renders — no flash of the default theme. Kept tiny
 * and dependency-free on purpose; it re-validates the cookie against the
 * allow-list so a tampered cookie can only ever yield a known theme id.
 */
export const THEME_INIT_SCRIPT = `(function(){try{var m=document.cookie.match(/(?:^|;\\s*)${THEME_COOKIE}=([^;]*)/);var t=m?decodeURIComponent(m[1]):null;var allowed=${JSON.stringify(APP_THEME_IDS)};document.documentElement.setAttribute("data-app-theme",allowed.indexOf(t)>-1?t:"${DEFAULT_THEME}")}catch(e){document.documentElement.setAttribute("data-app-theme","${DEFAULT_THEME}")}})()`;
