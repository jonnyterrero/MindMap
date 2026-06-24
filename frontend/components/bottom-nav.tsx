"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  BookOpen,
  Lightbulb,
  User,
  CalendarHeart,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

type Tab = {
  href: string;
  label: string;
  icon: LucideIcon;
};

// Five primary destinations. Check-In sits in the center as the emphasized
// daily habit action. Secondary routes live in the top "More" menu.
const leftTabs: Tab[] = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/journal", label: "Journal", icon: BookOpen },
];
const rightTabs: Tab[] = [
  { href: "/insights", label: "Insights", icon: Lightbulb },
  { href: "/settings", label: "Profile", icon: User },
];
const checkIn = { href: "/today", label: "Check-In", icon: CalendarHeart };

function isActive(pathname: string, href: string) {
  return pathname === href || pathname.startsWith(href + "/");
}

function TabLink({ tab, active }: { tab: Tab; active: boolean }) {
  const Icon = tab.icon;
  return (
    <Link
      href={tab.href}
      aria-current={active ? "page" : undefined}
      className={cn(
        "group flex flex-1 flex-col items-center justify-center gap-0.5 rounded-xl py-1.5 transition-colors",
        active ? "text-primary" : "text-muted-foreground hover:text-foreground",
      )}
    >
      <span
        className={cn(
          "flex h-9 w-9 items-center justify-center rounded-xl transition-all",
          active && "bg-primary/15",
        )}
      >
        <Icon className="h-5 w-5" aria-hidden="true" />
      </span>
      <span className="text-[10px] font-medium leading-none">{tab.label}</span>
    </Link>
  );
}

export function BottomNav() {
  const pathname = usePathname();
  const CheckInIcon = checkIn.icon;
  const checkInActive = isActive(pathname, checkIn.href);

  return (
    <nav
      aria-label="Primary"
      className="fixed inset-x-0 bottom-0 z-50 md:hidden safe-area-bottom"
    >
      <div className="mx-auto mb-2 flex max-w-md items-stretch gap-1 px-3">
        <div className="glass-dock relative flex flex-1 items-center rounded-[28px] px-2 py-1.5">
          {leftTabs.map((tab) => (
            <TabLink key={tab.href} tab={tab} active={isActive(pathname, tab.href)} />
          ))}

          {/* Center emphasized Check-In action */}
          <div className="flex flex-1 justify-center">
            <Link
              href={checkIn.href}
              aria-current={checkInActive ? "page" : undefined}
              aria-label={checkIn.label}
              className="group -mt-7 flex flex-col items-center gap-0.5"
            >
              <span
                className={cn(
                  "flex h-14 w-14 items-center justify-center rounded-full border border-white/40 bg-primary text-primary-foreground shadow-lg transition-all",
                  "ring-4 ring-background/60 group-hover:brightness-110 group-active:scale-95",
                  checkInActive && "ring-primary/30",
                )}
              >
                <CheckInIcon className="h-6 w-6" aria-hidden="true" />
              </span>
              <span
                className={cn(
                  "text-[10px] font-semibold leading-none",
                  checkInActive ? "text-primary" : "text-foreground",
                )}
              >
                {checkIn.label}
              </span>
            </Link>
          </div>

          {rightTabs.map((tab) => (
            <TabLink key={tab.href} tab={tab} active={isActive(pathname, tab.href)} />
          ))}
        </div>
      </div>
    </nav>
  );
}
