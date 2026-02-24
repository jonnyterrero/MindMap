"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Brain, CalendarCheck, ListChecks, BarChart3, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { signout } from "@/app/auth/actions";
import { cn } from "@/lib/utils";
import type { User } from "@supabase/supabase-js";

const navItems = [
  { href: "/today", label: "Today", icon: CalendarCheck },
  { href: "/routines", label: "Routines", icon: ListChecks },
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
];

export function AppNav({ user }: { user: User }) {
  const pathname = usePathname();

  const displayName =
    user.user_metadata?.display_name || user.email?.split("@")[0] || "User";

  return (
    <header className="sticky top-0 z-50 glass-strong">
      <div className="container mx-auto max-w-4xl flex items-center justify-between h-14 px-4">
        <Link href="/today" className="flex items-center gap-2 font-semibold">
          <Brain className="h-5 w-5 text-primary" />
          <span className="hidden sm:inline">MindMap</span>
        </Link>

        <nav className="flex items-center gap-1">
          {navItems.map(({ href, label, icon: Icon }) => (
            <Link key={href} href={href}>
              <Button
                variant={pathname.startsWith(href) ? "default" : "ghost"}
                size="sm"
                className={cn(
                  "gap-1.5",
                  pathname.startsWith(href) && "pointer-events-none"
                )}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden sm:inline">{label}</span>
              </Button>
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground hidden sm:inline">
            {displayName}
          </span>
          <form action={signout}>
            <Button type="submit" variant="ghost" size="icon">
              <LogOut className="h-4 w-4" />
            </Button>
          </form>
        </div>
      </div>
    </header>
  );
}
