"use client"

import { Moon, Clock, Star } from "lucide-react"
import { useHealthData } from "@/contexts/HealthDataContext"

export function SleepWidget() {
  const { sleepData } = useHealthData()

  const last7Days = sleepData.slice(0, 7).reverse()

  if (last7Days.length === 0) {
    return (
      <div className="glass rounded-[var(--radius-lg)] p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-primary-dark)] flex items-center justify-center">
              <Moon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-[var(--color-text)]">Sleep Tracking</h3>
              <p className="text-sm text-[var(--color-text-secondary)]">Last 7 days</p>
            </div>
          </div>
        </div>
        <div className="text-center py-12">
          <p className="text-[var(--color-text-secondary)] mb-4">No sleep data yet</p>
          <p className="text-sm text-[var(--color-text-muted)]">Start logging your sleep to see insights here</p>
        </div>
      </div>
    )
  }

  const maxHours = Math.max(...last7Days.map((d) => d.hours), 8)

  const avgHours = (last7Days.reduce((sum, d) => sum + d.hours, 0) / last7Days.length).toFixed(1)
  const avgQuality = (last7Days.reduce((sum, d) => sum + d.quality, 0) / last7Days.length).toFixed(1)

  return (
    <div className="glass rounded-[var(--radius-lg)] p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-primary-dark)] flex items-center justify-center">
            <Moon className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-[var(--color-text)]">Sleep Tracking</h3>
            <p className="text-sm text-[var(--color-text-secondary)]">Last {last7Days.length} days</p>
          </div>
        </div>
      </div>

      {/* Sleep Chart */}
      <div className="space-y-4 mb-6">
        {last7Days.map((data, index) => {
          const date = new Date(data.date)
          const dayName = date.toLocaleDateString("en-US", { weekday: "short" })

          return (
            <div key={index} className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-[var(--color-text-secondary)] font-medium w-12">{dayName}</span>
                <div className="flex-1 mx-4">
                  <div className="h-8 bg-white/50 rounded-[var(--radius-sm)] overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-[var(--color-primary-light)] to-[var(--color-primary)] rounded-[var(--radius-sm)] flex items-center justify-end px-3"
                      style={{ width: `${(data.hours / maxHours) * 100}%` }}
                    >
                      <span className="text-xs font-bold text-white">{data.hours}h</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                  <span className="text-[var(--color-text)] font-medium w-8">{data.quality}</span>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 gap-4 pt-4 border-t border-[var(--color-border)]">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-[var(--radius-sm)] bg-[var(--color-glass-purple)] flex items-center justify-center">
            <Clock className="w-5 h-5 text-[var(--color-primary)]" />
          </div>
          <div>
            <p className="text-xs text-[var(--color-text-secondary)]">Avg Duration</p>
            <p className="text-lg font-bold text-[var(--color-text)]">{avgHours}h</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-[var(--radius-sm)] bg-[var(--color-glass-pink)] flex items-center justify-center">
            <Star className="w-5 h-5 text-[var(--color-secondary-dark)]" />
          </div>
          <div>
            <p className="text-xs text-[var(--color-text-secondary)]">Avg Quality</p>
            <p className="text-lg font-bold text-[var(--color-text)]">{avgQuality}/10</p>
          </div>
        </div>
      </div>
    </div>
  )
}
