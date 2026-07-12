"use client"

import type React from "react"

import { useState } from "react"
import { Activity, X, Calendar } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useHealthData } from "@/contexts/HealthDataContext"
import { toast } from "sonner"

interface SymptomsLogFormProps {
  onClose: () => void
}

export function SymptomsLogForm({ onClose }: SymptomsLogFormProps) {
  const [logDate, setLogDate] = useState(new Date().toISOString().split("T")[0])
  const [giFlare, setGiFlare] = useState(0)
  const [skinFlare, setSkinFlare] = useState(0)
  const [migraine, setMigraine] = useState(0)
  const [notes, setNotes] = useState("")
  const { addSymptomEntry } = useHealthData()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    addSymptomEntry({
      date: logDate,
      gi: giFlare,
      skin: skinFlare,
      migraine: migraine,
    })

    toast.success("Symptoms logged successfully!", {
      description: `GI: ${giFlare}, Skin: ${skinFlare}, Migraine: ${migraine}`,
    })

    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="glass rounded-[var(--radius-lg)] p-6 w-full max-w-md max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-[var(--color-text)]">Log Symptoms</h2>
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
              className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-orange-400"
            />
          </div>

          {/* GI Flare */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-[var(--color-text)]">GI Flare Severity</label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="0"
                max="10"
                value={giFlare}
                onChange={(e) => setGiFlare(Number(e.target.value))}
                className="flex-1 h-2 bg-white/60 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-orange-400 [&::-webkit-slider-thumb]:to-red-500"
              />
              <span className="text-2xl font-bold text-[var(--color-text)] w-12 text-center">{giFlare}</span>
            </div>
            <div className="flex justify-between text-xs text-[var(--color-text-secondary)]">
              <span>None</span>
              <span>Severe</span>
            </div>
          </div>

          {/* Skin Flare */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-[var(--color-text)]">Skin Flare Severity</label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="0"
                max="10"
                value={skinFlare}
                onChange={(e) => setSkinFlare(Number(e.target.value))}
                className="flex-1 h-2 bg-white/60 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-pink-400 [&::-webkit-slider-thumb]:to-pink-600"
              />
              <span className="text-2xl font-bold text-[var(--color-text)] w-12 text-center">{skinFlare}</span>
            </div>
            <div className="flex justify-between text-xs text-[var(--color-text-secondary)]">
              <span>None</span>
              <span>Severe</span>
            </div>
          </div>

          {/* Migraine */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-[var(--color-text)]">Migraine Intensity</label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="0"
                max="10"
                value={migraine}
                onChange={(e) => setMigraine(Number(e.target.value))}
                className="flex-1 h-2 bg-white/60 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-purple-400 [&::-webkit-slider-thumb]:to-purple-600"
              />
              <span className="text-2xl font-bold text-[var(--color-text)] w-12 text-center">{migraine}</span>
            </div>
            <div className="flex justify-between text-xs text-[var(--color-text-secondary)]">
              <span>None</span>
              <span>Severe</span>
            </div>
          </div>

          {/* Notes */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-[var(--color-text)]">Additional Notes (Optional)</label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Any triggers, medications, or observations..."
              rows={3}
              className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] placeholder:text-[var(--color-text-muted)] focus:outline-none focus:ring-2 focus:ring-orange-400 resize-none"
            />
          </div>

          {/* Submit Button */}
          <Button
            type="submit"
            className="w-full py-3 bg-gradient-to-r from-orange-400 to-red-500 text-white font-semibold rounded-[var(--radius)] hover:opacity-90 transition-opacity"
          >
            Save Symptoms Log
          </Button>
        </form>
      </div>
    </div>
  )
}
