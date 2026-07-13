"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Brain, Home, CalendarCheck, ListChecks, Pill, BookOpen,
  BarChart3, Target, Heart, Lightbulb, MoreHorizontal,
  Settings, LogOut, Activity, MessageCircle, FileText, Network,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { signout } from "@/app/auth/actions";
import { SyncIndicator } from "@/components/sync-indicator";
import { cn } from "@/lib/utils";
import type { User } from "@supabase/supabase-js";

const primaryNav = [
  { href: "/home", label: "Home", icon: Home },
  { href: "/today", label: "Today", icon: CalendarCheck },
  { href: "/journal", label: "Journal", icon: BookOpen },
  { href: "/medications", label: "Meds", icon: Pill },
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
];

const moreNav = [
  { href: "/companion", label: "Companion", icon: MessageCircle },
  { href: "/insights", label: "Insights", icon: Lightbulb },
  { href: "/mindmap", label: "Mindmap", icon: Network },
  { href: "/reports", label: "Reports", icon: FileText },
  { href: "/routines", label: "Routines", icon: ListChecks },
  { href: "/body-map", label: "Body map", icon: Activity },
  { href: "/goals", label: "Goals", icon: Target },
  { href: "/therapy", label: "Therapy", icon: Heart },
];

export function AppNav({ user }: { user: User }) {
  const pathname = usePathname();

  const displayName =
    user.user_metadata?.display_name || user.email?.split("@")[0] || "User";

  const isMoreActive = moreNav.some((item) => pathname.startsWith(item.href));

  return (
    <header className="sticky top-0 z-50 glass-strong safe-area-top">
      <div className="container mx-auto max-w-4xl flex items-center justify-between h-14 px-4">
        <div className="flex items-center gap-2">
          <Link href="/home" className="flex items-center gap-2 font-semibold">
            <Brain className="h-5 w-5 text-primary" />
            <span className="hidden sm:inline">MindMap</span>
          </Link>
          <SyncIndicator />
        </div>

        <nav className="flex items-center gap-1">
          {primaryNav.map(({ href, label, icon: Icon }) => (
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

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant={isMoreActive ? "default" : "ghost"}
                size="sm"
                className="gap-1.5"
              >
                <MoreHorizontal className="h-4 w-4" />
                <span className="hidden sm:inline">More</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {moreNav.map(({ href, label, icon: Icon }) => (
                <DropdownMenuItem key={href} asChild>
                  <Link
                    href={href}
                    className={cn(
                      "flex items-center gap-2",
                      pathname.startsWith(href) && "font-medium"
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    {label}
                  </Link>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </nav>

        <div className="flex items-center gap-1">
          <span className="text-sm text-muted-foreground hidden sm:inline mr-1">
            {displayName}
          </span>
          <Link href="/settings">
            <Button variant="ghost" size="icon">
              <Settings className="h-4 w-4" />
            </Button>
          </Link>
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
