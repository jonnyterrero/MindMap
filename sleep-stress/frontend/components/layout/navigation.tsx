"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Home, Sparkles, TrendingUp, Award, User } from "lucide-react"

export function Navigation() {
  const pathname = usePathname()

  const links = [
    { href: "/", label: "Dashboard", icon: <Home className="w-5 h-5" /> },
    { href: "/insights", label: "Insights", icon: <Sparkles className="w-5 h-5" /> },
    { href: "/trends", label: "Trends", icon: <TrendingUp className="w-5 h-5" /> },
    { href: "/achievements", label: "Achievements", icon: <Award className="w-5 h-5" /> },
    { href: "/profile", label: "Profile", icon: <User className="w-5 h-5" /> },
  ]

  return (
    <nav className="fixed bottom-0 left-0 right-0 md:top-0 md:bottom-auto z-40">
      <div className="glass border-t md:border-t-0 md:border-b border-[var(--color-border)]">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-around md:justify-start md:gap-8 py-3">
            {links.map((link) => {
              const isActive = pathname === link.href
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex flex-col md:flex-row items-center gap-1 md:gap-2 px-3 py-2 rounded-[var(--radius)] transition-all ${
                    isActive
                      ? "bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] text-white"
                      : "text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-white/50"
                  }`}
                >
                  {link.icon}
                  <span className="text-xs md:text-sm font-medium">{link.label}</span>
                </Link>
              )
            })}
          </div>
        </div>
      </div>
    </nav>
  )
}
