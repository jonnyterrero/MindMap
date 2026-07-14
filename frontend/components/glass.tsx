import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cn } from "@/lib/utils";

/**
 * Liquid-glass design-system primitives.
 *
 * GlassPanel  — frosted surface for cards, sections, docks.
 * GlassButton — frosted action button; `glow` emphasizes a primary action.
 *
 * Colors come from theme tokens, so these restyle automatically when the
 * user switches their color mood in Settings → Appearance.
 */

export function GlassPanel({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("glass-panel rounded-2xl", className)}
      {...props}
    />
  );
}

export const GlassButton = React.forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    asChild?: boolean;
    glow?: boolean;
  }
>(({ className, asChild = false, glow = false, ...props }, ref) => {
  const Comp = asChild ? Slot : "button";
  return (
    <Comp
      ref={ref}
      className={cn(
        "inline-flex min-h-11 items-center justify-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium",
        "backdrop-blur-xl border transition-all duration-200",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        "disabled:pointer-events-none disabled:opacity-50",
        glow
          ? "bg-primary text-primary-foreground border-white/30 shadow-lg hover:brightness-110 active:scale-[0.98]"
          : "bg-white/55 text-foreground border-white/40 shadow-sm hover:bg-white/70 active:scale-[0.98]",
        className,
      )}
      {...props}
    />
  );
});
GlassButton.displayName = "GlassButton";
