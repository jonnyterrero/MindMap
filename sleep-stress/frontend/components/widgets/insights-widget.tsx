"use client"

import { Sparkles, TrendingUp, AlertCircle, ArrowRight } from "lucide-react"
import Link from "next/link"

export function InsightsWidget() {
  const insights = [
    {
      type: "positive",
      icon: <TrendingUp className="w-4 h-4" />,
      title: "Sleep Improving",
      description: "Your sleep quality increased by 15% this week",
      color: "from-green-400 to-emerald-500",
    },
    {
      type: "warning",
      icon: <AlertCircle className="w-4 h-4" />,
      title: "High Stress Alert",
      description: "Stress levels above 7 for 3 consecutive days",
      color: "from-orange-400 to-red-500",
    },
  ]

  return (
    <div className="glass rounded-[var(--radius-lg)] p-6">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-5 h-5 text-[var(--color-primary)]" />
        <h3 className="text-lg font-bold text-[var(--color-text)]">AI Insights</h3>
      </div>

      <div className="space-y-3 mb-4">
        {insights.map((insight, index) => (
          <div
            key={index}
            className="glass-hover rounded-[var(--radius)] p-4 border-l-4"
            style={{
              borderLeftColor: insight.type === "positive" ? "var(--color-success)" : "var(--color-warning)",
            }}
          >
            <div className="flex items-start gap-3">
              <div
                className={`w-8 h-8 rounded-[var(--radius-sm)] bg-gradient-to-br ${insight.color} flex items-center justify-center text-white`}
              >
                {insight.icon}
              </div>
              <div className="flex-1">
                <p className="font-semibold text-[var(--color-text)] text-sm mb-1">{insight.title}</p>
                <p className="text-xs text-[var(--color-text-secondary)] leading-relaxed">{insight.description}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <Link
        href="/insights"
        className="w-full flex items-center justify-center gap-2 py-2 text-sm font-medium text-[var(--color-primary)] hover:text-[var(--color-primary-dark)] transition-colors"
      >
        View All Insights
        <ArrowRight className="w-4 h-4" />
      </Link>
    </div>
  )
}
