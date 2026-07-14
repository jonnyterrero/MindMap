import type React from "react"
import { TrendingUp, TrendingDown } from "lucide-react"

interface StatsWidgetProps {
  icon: React.ReactNode
  label: string
  value: string
  unit: string
  trend: string
  color: "primary" | "secondary" | "accent" | "success"
}

export function StatsWidget({ icon, label, value, unit, trend, color }: StatsWidgetProps) {
  const isPositive = trend.startsWith("+")

  const colorClasses = {
    primary: "text-[var(--color-primary)]",
    secondary: "text-[var(--color-secondary-dark)]",
    accent: "text-[var(--color-accent-dark)]",
    success: "text-green-500",
  }

  return (
    <div className="glass glass-hover rounded-[var(--radius-lg)] p-6">
      <div className="flex items-start justify-between mb-4">
        <div className={`${colorClasses[color]}`}>{icon}</div>
        <div className={`flex items-center gap-1 text-sm ${isPositive ? "text-green-500" : "text-red-500"}`}>
          {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
          <span className="font-medium">{trend}</span>
        </div>
      </div>
      <div className="space-y-1">
        <p className="text-sm text-[var(--color-text-secondary)]">{label}</p>
        <p className="text-3xl font-bold text-[var(--color-text)]">
          {value}
          <span className="text-lg text-[var(--color-text-muted)]">{unit}</span>
        </p>
      </div>
    </div>
  )
}
