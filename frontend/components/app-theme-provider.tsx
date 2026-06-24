"use client";

import {
  createContext,
  useContext,
  useState,
  useTransition,
  type ReactNode,
} from "react";
import { cn } from "@/lib/utils";
import {
  DEFAULT_THEME,
  normalizeTheme,
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

/**
 * Provides the active app color theme to the tree and renders the wrapper
 * element that carries `data-app-theme`. Because the design tokens are scoped
 * to that attribute (see globals.css), changing the theme restyles every
 * descendant instantly. The selection is persisted to the user's profile in
 * the background.
 */
export function AppThemeProvider({
  initialTheme,
  children,
  className,
  persist = true,
}: {
  initialTheme?: AppThemeId | string | null;
  children: ReactNode;
  className?: string;
  /** When false, selections stay local (e.g. on the public landing page). */
  persist?: boolean;
}) {
  const [theme, setThemeState] = useState<AppThemeId>(
    normalizeTheme(initialTheme ?? DEFAULT_THEME),
  );
  const [pending, startTransition] = useTransition();

  function setTheme(next: AppThemeId) {
    setThemeState(next); // optimistic — instant visual feedback
    if (!persist) return;
    startTransition(async () => {
      await updateAppTheme(next);
    });
  }

  return (
    <AppThemeContext.Provider value={{ theme, setTheme, pending }}>
      <div
        data-app-theme={theme}
        className={cn("app-bg min-h-screen", className)}
      >
        {children}
      </div>
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
