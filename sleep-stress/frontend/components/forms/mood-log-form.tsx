"use client"

import type React from "react"

import { useState } from "react"
import { Heart, X, Smile, Frown, Meh, Calendar } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useHealthData } from "@/contexts/HealthDataContext"
import { toast } from "sonner"

interface MoodLogFormProps {
  onClose: () => void
}

export function MoodLogForm({ onClose }: MoodLogFormProps) {
  const [logDate, setLogDate] = useState(new Date().toISOString().split("T")[0])
  const [moodScore, setMoodScore] = useState(7)
  const [stressScore, setStressScore] = useState(5)
  const [journal, setJournal] = useState("")
  const { addMoodEntry } = useHealthData()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    addMoodEntry({
      date: logDate,
      mood: moodScore,
      stress: stressScore,
    })

    toast.success("Mood logged successfully!", {
      description: `Mood: ${moodScore}/10, Stress: ${stressScore}/10`,
    })

    onClose()
  }

  const getMoodIcon = (score: number) => {
    if (score >= 7) return <Smile className="w-6 h-6" />
    if (score >= 4) return <Meh className="w-6 h-6" />
    return <Frown className="w-6 h-6" />
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="glass rounded-[var(--radius-lg)] p-6 w-full max-w-md max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-secondary)] to-[var(--color-secondary-dark)] flex items-center justify-center">
              <Heart className="w-6 h-6 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-[var(--color-text)]">Log Mood</h2>
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
              className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-secondary)]"
            />
          </div>

          {/* Mood Score */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-[var(--color-text)] flex items-center gap-2">
              {getMoodIcon(moodScore)}
              Mood Score
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="1"
                max="10"
                value={moodScore}
                onChange={(e) => setMoodScore(Number(e.target.value))}
                className="flex-1 h-2 bg-white/60 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-[var(--color-secondary)] [&::-webkit-slider-thumb]:to-[var(--color-secondary-dark)]"
              />
              <span className="text-2xl font-bold text-[var(--color-text)] w-12 text-center">{moodScore}</span>
            </div>
            <div className="flex justify-between text-xs text-[var(--color-text-secondary)]">
              <span>Very Low</span>
              <span>Very High</span>
            </div>
          </div>

          {/* Stress Score */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-[var(--color-text)]">Stress Level</label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="1"
                max="10"
                value={stressScore}
                onChange={(e) => setStressScore(Number(e.target.value))}
                className="flex-1 h-2 bg-white/60 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-[var(--color-accent)] [&::-webkit-slider-thumb]:to-[var(--color-accent-dark)]"
              />
              <span className="text-2xl font-bold text-[var(--color-text)] w-12 text-center">{stressScore}</span>
            </div>
            <div className="flex justify-between text-xs text-[var(--color-text-secondary)]">
              <span>Relaxed</span>
              <span>Very Stressed</span>
            </div>
          </div>

          {/* Journal Entry */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-[var(--color-text)]">Journal Entry (Optional)</label>
            <textarea
              value={journal}
              onChange={(e) => setJournal(e.target.value)}
              placeholder="How are you feeling today? What's on your mind?"
              rows={4}
              className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] placeholder:text-[var(--color-text-muted)] focus:outline-none focus:ring-2 focus:ring-[var(--color-secondary)] resize-none"
            />
          </div>

          {/* Submit Button */}
          <Button
            type="submit"
            className="w-full py-3 bg-gradient-to-r from-[var(--color-secondary)] to-[var(--color-accent)] text-white font-semibold rounded-[var(--radius)] hover:opacity-90 transition-opacity"
          >
            Save Mood Log
          </Button>
        </form>
      </div>
    </div>
  )
}
