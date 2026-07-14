"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  useTransition,
  type ReactNode,
} from "react";
import {
  DEFAULT_THEME,
  normalizeTheme,
  THEME_COOKIE,
  THEME_COOKIE_MAX_AGE,
  type AppThemeId,
} from "@/lib/themes";
import { updateAppTheme } from "@/app/(app)/settings/actions";

type ThemeContextValue = {
  theme: AppThemeId;
  setTheme: (theme: AppThemeId) => void;
  /** True while the selection is being persisted to the profile. */
  pending: boolean;
};

const AppThemeContext = createContext<ThemeContextValue | null>(null);

/** Keep <html data-app-theme> and the rendering-hint cookie in step. */
function applyTheme(theme: AppThemeId) {
  document.documentElement.setAttribute("data-app-theme", theme);
  document.cookie = `${THEME_COOKIE}=${theme}; path=/; max-age=${THEME_COOKIE_MAX_AGE}; samesite=lax`;
}

/**
 * Owns the active color theme for the authenticated app.
 *
 * The attribute is written to <html>, not to a wrapper element. That matters
 * for two reasons: Radix portals its dialogs, dropdowns, selects and popovers
 * onto <body>, and <body> is what paints the themed gradient — a wrapper div
 * would reach neither, leaving every modal and the page background stuck on
 * the default palette.
 *
 * On first paint the no-flash script in the root layout has already set the
 * attribute from the cookie. This provider reconciles that against the profile
 * (the cross-device source of truth) and takes over on change.
 */
export function AppThemeProvider({
  initialTheme,
  children,
  persist = true,
}: {
  initialTheme?: AppThemeId | string | null;
  children: ReactNode;
  /** When false, selections stay local and are never written to the profile. */
  persist?: boolean;
}) {
  const [theme, setThemeState] = useState<AppThemeId>(
    normalizeTheme(initialTheme ?? DEFAULT_THEME),
  );
  const [pending, startTransition] = useTransition();

  // Reconciles the cookie-driven first paint against the profile. The two
  // agree on every load except the first one on a new device, where no cookie
  // exists yet and the script fell back to the default.
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  function setTheme(next: AppThemeId) {
    setThemeState(next); // optimistic — the effect repaints immediately
    if (!persist) return;
    startTransition(async () => {
      await updateAppTheme(next);
    });
  }

  return (
    <AppThemeContext.Provider value={{ theme, setTheme, pending }}>
      {children}
    </AppThemeContext.Provider>
  );
}

export function useAppTheme() {
  const ctx = useContext(AppThemeContext);
  if (!ctx) {
    throw new Error("useAppTheme must be used within an AppThemeProvider");
  }
  return ctx;
}
