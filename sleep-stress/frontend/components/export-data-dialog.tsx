"use client"

import { useState } from "react"
import { Download, FileText, Database, X } from "lucide-react"
import {
  exportSleepData,
  exportMoodData,
  exportSymptomsData,
  exportProfileData,
  exportAllData,
  type SleepEntry,
  type MoodEntry,
  type SymptomEntry,
  type ProfileData,
  type GamificationData,
  type HealthGoal,
} from "@/lib/data-export"

interface ExportDataDialogProps {
  isOpen: boolean
  onClose: () => void
}

export function ExportDataDialog({ isOpen, onClose }: ExportDataDialogProps) {
  const [selectedData, setSelectedData] = useState({
    sleep: true,
    mood: true,
    symptoms: true,
    profile: true,
    gamification: false,
    goals: false,
  })
  const [format, setFormat] = useState<"csv" | "json">("csv")
  const [dateRange, setDateRange] = useState("30")

  if (!isOpen) return null

  // Mock data - in a real app, this would come from your data store/API
  const getSleepData = (): SleepEntry[] => {
    const data: SleepEntry[] = []
    for (let i = 0; i < Number.parseInt(dateRange); i++) {
      const date = new Date()
      date.setDate(date.getDate() - i)
      data.push({
        date: date.toISOString().split("T")[0],
        hours: 7 + Math.random() * 2,
        quality: 6 + Math.floor(Math.random() * 4),
      })
    }
    return data.reverse()
  }

  const getMoodData = (): MoodEntry[] => {
    const data: MoodEntry[] = []
    for (let i = 0; i < Number.parseInt(dateRange); i++) {
      const date = new Date()
      date.setDate(date.getDate() - i)
      data.push({
        date: date.toISOString().split("T")[0],
        mood: 6 + Math.floor(Math.random() * 4),
        stress: 3 + Math.floor(Math.random() * 4),
      })
    }
    return data.reverse()
  }

  const getSymptomsData = (): SymptomEntry[] => {
    const data: SymptomEntry[] = []
    for (let i = 0; i < Number.parseInt(dateRange); i++) {
      const date = new Date()
      date.setDate(date.getDate() - i)
      data.push({
        date: date.toISOString().split("T")[0],
        gi: 2 + Math.floor(Math.random() * 3),
        skin: 3 + Math.floor(Math.random() * 3),
        migraine: 1 + Math.floor(Math.random() * 3),
      })
    }
    return data.reverse()
  }

  const getProfileData = (): ProfileData => ({
    name: "Sarah Johnson",
    email: "sarah.johnson@example.com",
    age: 28,
    gender: "Female",
    height: 165,
    weight: 62,
    bmi: 22.8,
    bmiCategory: "Normal weight",
    conditions: ["Anxiety", "IBS", "Migraine"],
    medications: ["Anxiety medication", "Probiotics"],
    allergies: ["Lactose intolerance"],
    activityLevel: "Moderate",
  })

  const getGamificationData = (): GamificationData => ({
    level: 8,
    xp: 2450,
    xpToNextLevel: 3000,
    currentStreak: 12,
    longestStreak: 28,
    badges: [
      { name: "Early Bird", earned: true, progress: 100 },
      { name: "Consistency King", earned: true, progress: 100 },
      { name: "Mood Master", earned: false, progress: 75 },
    ],
    achievements: [
      { name: "7-Day Streak", description: "Logged data for 7 consecutive days", date: "2025-01-15" },
      { name: "Sleep Champion", description: "Achieved 8+ hours of sleep for 5 days", date: "2025-01-20" },
    ],
  })

  const getHealthGoals = (): HealthGoal[] => [
    { id: 1, name: "Sleep 7.5 hours daily", progress: 78, target: 100 },
    { id: 2, name: "Reduce stress below 5", progress: 45, target: 100 },
    { id: 3, name: "Maintain mood above 7", progress: 92, target: 100 },
  ]

  const handleExport = () => {
    const sleepData = getSleepData()
    const moodData = getMoodData()
    const symptomsData = getSymptomsData()
    const profileData = getProfileData()
    const gamificationData = getGamificationData()
    const healthGoals = getHealthGoals()

    // If all data types are selected and format is JSON, export as single file
    const allSelected = Object.values(selectedData).every((v) => v)
    if (allSelected && format === "json") {
      exportAllData(sleepData, moodData, symptomsData, profileData, gamificationData, healthGoals, "json")
      onClose()
      return
    }

    // Export individual data types
    if (selectedData.sleep) {
      exportSleepData(sleepData, format)
    }
    if (selectedData.mood) {
      exportMoodData(moodData, format)
    }
    if (selectedData.symptoms) {
      exportSymptomsData(symptomsData, format)
    }
    if (selectedData.profile) {
      exportProfileData(profileData, format)
    }
    if (selectedData.gamification && format === "json") {
      const timestamp = new Date().toISOString().split("T")[0]
      const json = JSON.stringify(gamificationData, null, 2)
      const blob = new Blob([json], { type: "application/json" })
      const url = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = `gamification-data-${timestamp}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
    }
    if (selectedData.goals && format === "json") {
      const timestamp = new Date().toISOString().split("T")[0]
      const json = JSON.stringify(healthGoals, null, 2)
      const blob = new Blob([json], { type: "application/json" })
      const url = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = `health-goals-${timestamp}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
    }

    onClose()
  }

  const toggleDataType = (type: keyof typeof selectedData) => {
    setSelectedData((prev) => ({ ...prev, [type]: !prev[type] }))
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="glass rounded-[var(--radius-lg)] p-6 max-w-lg w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-secondary)] flex items-center justify-center">
              <Download className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-[var(--color-text)]">Export Your Data</h2>
              <p className="text-sm text-[var(--color-text-secondary)]">Download your health data</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-[var(--radius-sm)] glass-hover flex items-center justify-center text-[var(--color-text-secondary)]"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Date Range Selector */}
        <div className="mb-6">
          <label className="text-sm font-semibold text-[var(--color-text)] mb-3 block">Date Range</label>
          <div className="grid grid-cols-4 gap-2">
            {["7", "30", "90", "365"].map((days) => (
              <button
                key={days}
                onClick={() => setDateRange(days)}
                className={`px-4 py-2 rounded-[var(--radius)] font-medium transition-all ${
                  dateRange === days
                    ? "bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] text-white"
                    : "glass-hover text-[var(--color-text)]"
                }`}
              >
                {days}d
              </button>
            ))}
          </div>
        </div>

        {/* Data Type Selection */}
        <div className="mb-6">
          <label className="text-sm font-semibold text-[var(--color-text)] mb-3 block">Select Data to Export</label>
          <div className="space-y-2">
            {[
              { key: "sleep", label: "Sleep Data", icon: "🌙" },
              { key: "mood", label: "Mood & Stress Data", icon: "❤️" },
              { key: "symptoms", label: "Symptoms Data", icon: "⚠️" },
              { key: "profile", label: "Profile Information", icon: "👤" },
              { key: "gamification", label: "Achievements & Badges", icon: "🏆" },
              { key: "goals", label: "Health Goals", icon: "🎯" },
            ].map((item) => (
              <button
                key={item.key}
                onClick={() => toggleDataType(item.key as keyof typeof selectedData)}
                className={`w-full p-3 rounded-[var(--radius)] flex items-center gap-3 transition-all ${
                  selectedData[item.key as keyof typeof selectedData]
                    ? "bg-gradient-to-r from-[var(--color-glass-blue)] to-[var(--color-glass-purple)] border-2 border-[var(--color-primary)]"
                    : "glass-hover"
                }`}
              >
                <span className="text-2xl">{item.icon}</span>
                <span className="text-sm font-medium text-[var(--color-text)]">{item.label}</span>
                <div className="ml-auto">
                  <div
                    className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                      selectedData[item.key as keyof typeof selectedData]
                        ? "bg-[var(--color-primary)] border-[var(--color-primary)]"
                        : "border-[var(--color-border)]"
                    }`}
                  >
                    {selectedData[item.key as keyof typeof selectedData] && (
                      <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Format Selection */}
        <div className="mb-6">
          <label className="text-sm font-semibold text-[var(--color-text)] mb-3 block">Export Format</label>
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => setFormat("csv")}
              className={`p-4 rounded-[var(--radius)] flex flex-col items-center gap-2 transition-all ${
                format === "csv"
                  ? "bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] text-white"
                  : "glass-hover text-[var(--color-text)]"
              }`}
            >
              <FileText className="w-6 h-6" />
              <span className="font-medium">CSV</span>
              <span className="text-xs opacity-80">Spreadsheet format</span>
            </button>
            <button
              onClick={() => setFormat("json")}
              className={`p-4 rounded-[var(--radius)] flex flex-col items-center gap-2 transition-all ${
                format === "json"
                  ? "bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] text-white"
                  : "glass-hover text-[var(--color-text)]"
              }`}
            >
              <Database className="w-6 h-6" />
              <span className="font-medium">JSON</span>
              <span className="text-xs opacity-80">Complete backup</span>
            </button>
          </div>
        </div>

        {/* Info Message */}
        {format === "csv" && (selectedData.gamification || selectedData.goals) && (
          <div className="mb-6 p-3 rounded-[var(--radius)] bg-yellow-100 border border-yellow-300">
            <p className="text-xs text-yellow-800">
              Note: Gamification and Goals data can only be exported in JSON format. These will be skipped for CSV
              export.
            </p>
          </div>
        )}

        {/* Export Button */}
        <button
          onClick={handleExport}
          disabled={!Object.values(selectedData).some((v) => v)}
          className="w-full py-3 rounded-[var(--radius)] bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] text-white font-semibold flex items-center justify-center gap-2 hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Download className="w-5 h-5" />
          Export Data
        </button>
      </div>
    </div>
  )
}
