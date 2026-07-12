"use client"

import type React from "react"

import { useState } from "react"
import { Moon, X, Clock, Star, Calendar } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useHealthData } from "@/contexts/HealthDataContext"
import { toast } from "sonner"

interface SleepLogFormProps {
  onClose: () => void
}

export function SleepLogForm({ onClose }: SleepLogFormProps) {
  const [logDate, setLogDate] = useState(new Date().toISOString().split("T")[0])
  const [sleepStart, setSleepStart] = useState("22:00")
  const [sleepEnd, setSleepEnd] = useState("07:00")
  const [quality, setQuality] = useState(7)
  const { addSleepEntry } = useHealthData()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    const [startHour, startMin] = sleepStart.split(":").map(Number)
    const [endHour, endMin] = sleepEnd.split(":").map(Number)

    let hours = endHour - startHour + (endMin - startMin) / 60
    if (hours < 0) hours += 24

    addSleepEntry({
      date: logDate,
      hours: Math.round(hours * 10) / 10,
      quality,
    })

    toast.success("Sleep logged successfully!", {
      description: `${hours.toFixed(1)} hours with quality ${quality}/10`,
    })

    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="glass rounded-[var(--radius-lg)] p-6 w-full max-w-md">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-primary-dark)] flex items-center justify-center">
              <Moon className="w-6 h-6 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-[var(--color-text)]">Log Sleep</h2>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-[var(--radius-sm)] hover:bg-white/50 flex items-center justify-center transition-colors"
          >
            <X className="w-5 h-5 text-[var(--color-text)]" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Date Picker */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-[var(--color-text)] flex items-center gap-2">
              <Calendar className="w-4 h-4" />
              Date
            </label>
            <input
              type="date"
              value={logDate}
              onChange={(e) => setLogDate(e.target.value)}
              max={new Date().toISOString().split("T")[0]}
              className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>

          {/* Sleep Start Time */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-[var(--color-text)] flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Sleep Start Time
            </label>
            <input
              type="time"
              value={sleepStart}
              onChange={(e) => setSleepStart(e.target.value)}
              className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>

          {/* Sleep End Time */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-[var(--color-text)] flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Wake Up Time
            </label>
            <input
              type="time"
              value={sleepEnd}
              onChange={(e) => setSleepEnd(e.target.value)}
              className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
            />
          </div>

          {/* Sleep Quality */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-[var(--color-text)] flex items-center gap-2">
              <Star className="w-4 h-4" />
              Sleep Quality
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="1"
                max="10"
                value={quality}
                onChange={(e) => setQuality(Number(e.target.value))}
                className="flex-1 h-2 bg-white/60 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-[var(--color-primary)] [&::-webkit-slider-thumb]:to-[var(--color-secondary)]"
              />
              <span className="text-2xl font-bold text-[var(--color-text)] w-12 text-center">{quality}</span>
            </div>
            <div className="flex justify-between text-xs text-[var(--color-text-secondary)]">
              <span>Poor</span>
              <span>Excellent</span>
            </div>
          </div>

          {/* Submit Button */}
          <Button
            type="submit"
            className="w-full py-3 bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] text-white font-semibold rounded-[var(--radius)] hover:opacity-90 transition-opacity"
          >
            Save Sleep Log
          </Button>
        </form>
      </div>
    </div>
  )
}
