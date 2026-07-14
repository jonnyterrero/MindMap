"use client"

import { Heart } from "lucide-react"
import { useHealthData } from "@/contexts/HealthDataContext"

export function MoodWidget() {
  const { moodData } = useHealthData()

  const recentData = moodData.slice(0, 6).reverse()

  if (recentData.length === 0) {
    return (
      <div className="glass rounded-[var(--radius-lg)] p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-secondary)] to-[var(--color-secondary-dark)] flex items-center justify-center">
              <Heart className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-[var(--color-text)]">Mood & Stress</h3>
              <p className="text-sm text-[var(--color-text-secondary)]">Recent entries</p>
            </div>
          </div>
        </div>
        <div className="text-center py-12">
          <p className="text-[var(--color-text-secondary)] mb-4">No mood data yet</p>
          <p className="text-sm text-[var(--color-text-muted)]">Start logging your mood to see patterns here</p>
        </div>
      </div>
    )
  }

  const maxValue = 10

  return (
    <div className="glass rounded-[var(--radius-lg)] p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-secondary)] to-[var(--color-secondary-dark)] flex items-center justify-center">
            <Heart className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-[var(--color-text)]">Mood & Stress</h3>
            <p className="text-sm text-[var(--color-text-secondary)]">Last {recentData.length} entries</p>
          </div>
        </div>
      </div>

      {/* Mood Chart */}
      <div className="relative h-48 mb-6">
        <div className="absolute inset-0 flex items-end justify-between gap-2">
          {recentData.map((data, index) => {
            const date = new Date(data.date)
            const label = date.toLocaleDateString("en-US", { month: "short", day: "numeric" })

            return (
              <div key={index} className="flex-1 flex flex-col items-center gap-2">
                <div className="w-full space-y-1">
                  {/* Mood Bar */}
                  <div
                    className="w-full bg-white/50 rounded-[var(--radius-sm)] overflow-hidden"
                    style={{ height: "80px" }}
                  >
                    <div
                      className="w-full bg-gradient-to-t from-[var(--color-secondary)] to-[var(--color-secondary-light)] rounded-[var(--radius-sm)]"
                      style={{ height: `${(data.mood / maxValue) * 100}%`, marginTop: "auto" }}
                    />
                  </div>
                  {/* Stress Bar */}
                  <div
                    className="w-full bg-white/50 rounded-[var(--radius-sm)] overflow-hidden"
                    style={{ height: "60px" }}
                  >
                    <div
                      className="w-full bg-gradient-to-t from-[var(--color-accent-dark)] to-[var(--color-accent)] rounded-[var(--radius-sm)]"
                      style={{ height: `${(data.stress / maxValue) * 100}%`, marginTop: "auto" }}
                    />
                  </div>
                </div>
                <span className="text-xs text-[var(--color-text-secondary)] font-medium">{label}</span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 pt-4 border-t border-[var(--color-border)]">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[var(--color-secondary)]" />
          <span className="text-sm text-[var(--color-text-secondary)]">Mood Score</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[var(--color-accent-dark)]" />
          <span className="text-sm text-[var(--color-text-secondary)]">Stress Level</span>
        </div>
      </div>
    </div>
  )
}
