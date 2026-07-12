"use client"

import { useState } from "react"
import { Plus, Moon, Heart, Activity } from "lucide-react"
import { SleepLogForm } from "@/components/forms/sleep-log-form"
import { MoodLogForm } from "@/components/forms/mood-log-form"
import { SymptomsLogForm } from "@/components/forms/symptoms-log-form"

export function QuickLogWidget() {
  const [showSleepForm, setShowSleepForm] = useState(false)
  const [showMoodForm, setShowMoodForm] = useState(false)
  const [showSymptomsForm, setShowSymptomsForm] = useState(false)

  return (
    <>
      <div className="glass rounded-[var(--radius-lg)] p-6">
        <h3 className="text-lg font-bold text-[var(--color-text)] mb-4">Quick Log</h3>

        <div className="space-y-3">
          <button
            onClick={() => setShowSleepForm(true)}
            className="w-full glass-hover bg-gradient-to-r from-[var(--color-glass-purple)] to-[var(--color-glass-blue)] rounded-[var(--radius)] p-4 flex items-center gap-3 text-left"
          >
            <div className="w-10 h-10 rounded-[var(--radius-sm)] bg-white/80 flex items-center justify-center">
              <Moon className="w-5 h-5 text-[var(--color-primary)]" />
            </div>
            <div className="flex-1">
              <p className="font-semibold text-[var(--color-text)]">Log Sleep</p>
              <p className="text-xs text-[var(--color-text-secondary)]">Track last night</p>
            </div>
            <Plus className="w-5 h-5 text-[var(--color-text-secondary)]" />
          </button>

          <button
            onClick={() => setShowMoodForm(true)}
            className="w-full glass-hover bg-gradient-to-r from-[var(--color-glass-pink)] to-[var(--color-glass-purple)] rounded-[var(--radius)] p-4 flex items-center gap-3 text-left"
          >
            <div className="w-10 h-10 rounded-[var(--radius-sm)] bg-white/80 flex items-center justify-center">
              <Heart className="w-5 h-5 text-[var(--color-secondary-dark)]" />
            </div>
            <div className="flex-1">
              <p className="font-semibold text-[var(--color-text)]">Log Mood</p>
              <p className="text-xs text-[var(--color-text-secondary)]">How are you feeling?</p>
            </div>
            <Plus className="w-5 h-5 text-[var(--color-text-secondary)]" />
          </button>

          <button
            onClick={() => setShowSymptomsForm(true)}
            className="w-full glass-hover bg-gradient-to-r from-[var(--color-glass-blue)] to-[var(--color-glass-pink)] rounded-[var(--radius)] p-4 flex items-center gap-3 text-left"
          >
            <div className="w-10 h-10 rounded-[var(--radius-sm)] bg-white/80 flex items-center justify-center">
              <Activity className="w-5 h-5 text-[var(--color-accent-dark)]" />
            </div>
            <div className="flex-1">
              <p className="font-semibold text-[var(--color-text)]">Log Symptoms</p>
              <p className="text-xs text-[var(--color-text-secondary)]">Track health changes</p>
            </div>
            <Plus className="w-5 h-5 text-[var(--color-text-secondary)]" />
          </button>
        </div>
      </div>

      {showSleepForm && <SleepLogForm onClose={() => setShowSleepForm(false)} />}
      {showMoodForm && <MoodLogForm onClose={() => setShowMoodForm(false)} />}
      {showSymptomsForm && <SymptomsLogForm onClose={() => setShowSymptomsForm(false)} />}
    </>
  )
}
