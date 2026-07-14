"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Check, Palette, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { THEME_LIST, type AppThemeId } from "@/lib/themes";
import { useAppTheme } from "@/components/app-theme-provider";

/**
 * Appearance section: pick a color "mood" for the workspace. Selecting a
 * theme updates the active nav color, primary CTA, selected states, and
 * gradient accents instantly, and persists the choice to the profile.
 */
export function AppearanceSettings() {
  const { theme, setTheme, pending } = useAppTheme();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <Palette className="h-5 w-5 text-primary" aria-hidden="true" />
          Appearance
          {pending && (
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" aria-hidden="true" />
          )}
        </CardTitle>
        <CardDescription>
          Choose a color mood for your MindMap workspace.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <fieldset>
          <legend className="sr-only">App color theme</legend>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {THEME_LIST.map((t) => (
              <ThemeCard
                key={t.id}
                id={t.id}
                name={t.name}
                label={t.label}
                gradient={t.gradient}
                selected={theme === t.id}
                onSelect={() => setTheme(t.id)}
              />
            ))}
          </div>
        </fieldset>
      </CardContent>
    </Card>
  );
}

function ThemeCard({
  id,
  name,
  label,
  gradient,
  selected,
  onSelect,
}: {
  id: AppThemeId;
  name: string;
  label: string;
  gradient: string;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      aria-label={`${name} — ${label}`}
      data-app-theme={id}
      className={cn(
        "group relative overflow-hidden rounded-2xl border p-3 text-left transition-all",
        "backdrop-blur-xl bg-white/55 hover:bg-white/70 active:scale-[0.99]",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        selected
          ? "border-primary ring-2 ring-primary/40 shadow-sm"
          : "border-white/40",
      )}
    >
      <span
        className={cn(
          "block h-12 w-full rounded-xl bg-gradient-to-br shadow-inner",
          gradient,
        )}
        aria-hidden="true"
      />
      <span className="mt-2 flex items-center justify-between gap-1">
        <span className="text-sm font-semibold text-foreground">{name}</span>
        {selected && (
          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground">
            <Check className="h-3 w-3" aria-hidden="true" />
          </span>
        )}
      </span>
      <span className="block text-xs text-muted-foreground">{label}</span>
    </button>
  );
}
