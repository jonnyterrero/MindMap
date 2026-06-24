import type { ReactNode } from "react";
import type { User } from "@supabase/supabase-js";
import { AppThemeProvider } from "@/components/app-theme-provider";
import { AppHeader } from "@/components/app-header";
import { BottomNav } from "@/components/bottom-nav";
import type { AppThemeId } from "@/lib/themes";

/**
 * Authenticated app shell: responsive navigation (top glass nav on desktop,
 * bottom dock on mobile), themed background, and the active color theme.
 * Adds bottom padding on mobile so content clears the floating dock.
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
      <AppHeader user={user} />
      <main className="container mx-auto max-w-5xl px-4 py-6 pb-28 md:pb-8 safe-area-bottom">
        {children}
      </main>
      <BottomNav />
    </AppThemeProvider>
  );
}
