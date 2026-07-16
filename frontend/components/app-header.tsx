"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Brain, Home, CalendarCheck, BookOpen, Pill, BarChart3,
  Lightbulb, MoreHorizontal, Settings, LogOut, Activity,
  MessageCircle, FileText, Target, Heart, ListChecks, User,
  type LucideIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { signout } from "@/app/auth/actions";
import { SyncIndicator } from "@/components/sync-indicator";
import { cn } from "@/lib/utils";
import type { User as SupabaseUser } from "@supabase/supabase-js";

type NavItem = { href: string; label: string; icon: LucideIcon };

// Full primary nav shown inline on desktop.
const desktopPrimary: NavItem[] = [
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { href: "/today", label: "Check-In", icon: CalendarCheck },
  { href: "/journal", label: "Journal", icon: BookOpen },
  { href: "/insights", label: "Insights", icon: Lightbulb },
];

// Secondary destinations — in "More" on every viewport.
const moreNav: NavItem[] = [
  { href: "/home", label: "Home", icon: Home },
  { href: "/medications", label: "Meds", icon: Pill },
  { href: "/companion", label: "Companion", icon: MessageCircle },
  { href: "/reports", label: "Reports", icon: FileText },
  { href: "/routines", label: "Routines", icon: ListChecks },
  { href: "/body-map", label: "Body map", icon: Activity },
  { href: "/goals", label: "Goals", icon: Target },
  { href: "/therapy", label: "Therapy", icon: Heart },
];

function isActive(pathname: string, href: string) {
  return pathname === href || pathname.startsWith(href + "/");
}

export function AppHeader({ user }: { user: SupabaseUser }) {
  const pathname = usePathname();
  const displayName =
    user.user_metadata?.display_name || user.email?.split("@")[0] || "User";
  const moreActive = moreNav.some((item) => isActive(pathname, item.href));

  return (
    <header className="sticky top-0 z-40 glass-dock safe-area-top">
      <div className="container mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <Link href="/dashboard" className="flex items-center gap-2 font-semibold">
            <Brain className="h-5 w-5 text-primary" aria-hidden="true" />
            <span className="font-[family-name:var(--font-space-grotesk)]">MindMap</span>
          </Link>
          <SyncIndicator />
        </div>

        {/* Desktop inline primary nav. Button renders the Link itself via
            asChild — nesting <button> inside <a> is invalid HTML that breaks
            assistive tech, and navigation should be links, not buttons. */}
        <nav className="hidden items-center gap-1 md:flex" aria-label="Primary">
          {desktopPrimary.map(({ href, label, icon: Icon }) => {
            const active = isActive(pathname, href);
            return (
              <Button
                key={href}
                asChild
                variant={active ? "default" : "ghost"}
                size="sm"
                className="gap-1.5"
              >
                <Link href={href} aria-current={active ? "page" : undefined}>
                  <Icon className="h-4 w-4" aria-hidden="true" />
                  {label}
                </Link>
              </Button>
            );
          })}
        </nav>

        <div className="flex items-center gap-1">
          <span className="mr-1 hidden text-sm text-muted-foreground lg:inline">
            {displayName}
          </span>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant={moreActive ? "default" : "ghost"}
                size="sm"
                className="gap-1.5"
              >
                <MoreHorizontal className="h-4 w-4" aria-hidden="true" />
                <span className="hidden sm:inline">More</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuLabel>Explore</DropdownMenuLabel>
              {moreNav.map(({ href, label, icon: Icon }) => (
                <DropdownMenuItem key={href} asChild>
                  <Link
                    href={href}
                    className={cn(
                      "flex items-center gap-2",
                      isActive(pathname, href) && "font-medium text-primary",
                    )}
                  >
                    <Icon className="h-4 w-4" aria-hidden="true" />
                    {label}
                  </Link>
                </DropdownMenuItem>
              ))}
              <DropdownMenuSeparator />
              <DropdownMenuItem asChild>
                <Link href="/settings" className="flex items-center gap-2">
                  <User className="h-4 w-4" aria-hidden="true" />
                  Profile
                </Link>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <Button
            asChild
            variant="ghost"
            size="icon"
            className="hidden sm:inline-flex"
          >
            <Link
              href="/settings"
              aria-label="Settings"
              aria-current={isActive(pathname, "/settings") ? "page" : undefined}
            >
              <Settings className="h-4 w-4" aria-hidden="true" />
            </Link>
          </Button>
          <form action={signout}>
            <Button type="submit" variant="ghost" size="icon" aria-label="Sign out">
              <LogOut className="h-4 w-4" aria-hidden="true" />
            </Button>
          </form>
        </div>
      </div>
    </header>
  );
}
