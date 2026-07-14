import type { ReactNode } from "react";
import type { User } from "@supabase/supabase-js";
import { AppThemeProvider } from "@/components/app-theme-provider";
import { AppHeader } from "@/components/app-header";
import { BottomNav } from "@/components/bottom-nav";
import type { AppThemeId } from "@/lib/themes";

/**
 * Authenticated app shell: responsive navigation (top glass nav on desktop,
 * bottom dock on mobile) and the active color theme.
 *
 * The themed gradient is painted by <body> (globals.css) off the <html>
 * data-app-theme attribute, so the shell deliberately renders no background
 * element of its own — a wrapper would double-paint the translucent gradient.
 *
 * `pb-28` keeps content clear of the floating dock on mobile.
 */
export function AppShell({
  user,
  initialTheme,
  children,
}: {
  user: User;
  initialTheme: AppThemeId | string | null;
  children: ReactNode;
}) {
  return (
    <AppThemeProvider initialTheme={initialTheme}>
      <a href="#main-content" className="skip-link">
        Skip to content
      </a>
      <AppHeader user={user} />
      <main
        id="main-content"
        tabIndex={-1}
        className="container mx-auto max-w-5xl px-4 py-6 pb-28 md:pb-8 safe-area-bottom"
      >
        {children}
      </main>
      <BottomNav />
    </AppThemeProvider>
  );
}
