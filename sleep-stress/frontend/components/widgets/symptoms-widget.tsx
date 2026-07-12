"use client"

import { AlertTriangle, TrendingDown, TrendingUp } from "lucide-react"
import { useHealthData } from "@/contexts/HealthDataContext"

export function SymptomsWidget() {
  const { symptomsData } = useHealthData()

  const recentData = symptomsData.slice(0, 7)

  const symptoms = [
    {
      name: "GI Flare",
      current: recentData.length > 0 ? recentData[0].gi : 0,
      predicted: recentData.length > 1 ? Math.round((recentData[0].gi + recentData[1].gi) / 2) : 0,
      risk: "low",
    },
    {
      name: "Skin Flare",
      current: recentData.length > 0 ? recentData[0].skin : 0,
      predicted: recentData.length > 1 ? Math.round((recentData[0].skin + recentData[1].skin) / 2) : 0,
      risk: "medium",
    },
    {
      name: "Migraine",
      current: recentData.length > 0 ? recentData[0].migraine : 0,
      predicted: recentData.length > 1 ? Math.round((recentData[0].migraine + recentData[1].migraine) / 2) : 0,
      risk: "low",
    },
  ]

  symptoms.forEach((symptom) => {
    if (symptom.current >= 7) symptom.risk = "high"
    else if (symptom.current >= 4) symptom.risk = "medium"
    else symptom.risk = "low"
  })

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "low":
        return "from-green-400 to-emerald-500"
      case "medium":
        return "from-yellow-400 to-orange-500"
      case "high":
        return "from-orange-500 to-red-600"
      default:
        return "from-gray-400 to-gray-500"
    }
  }

  if (recentData.length === 0) {
    return (
      <div className="glass rounded-[var(--radius-lg)] p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center">
            <AlertTriangle className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-[var(--color-text)]">Symptom Tracking</h3>
            <p className="text-sm text-[var(--color-text-secondary)]">Current & predicted</p>
          </div>
        </div>
        <div className="text-center py-12">
          <p className="text-[var(--color-text-secondary)] mb-4">No symptom data yet</p>
          <p className="text-sm text-[var(--color-text-muted)]">Start logging symptoms to track patterns</p>
        </div>
      </div>
    )
  }

  return (
    <div className="glass rounded-[var(--radius-lg)] p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center">
          <AlertTriangle className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-[var(--color-text)]">Symptom Tracking</h3>
          <p className="text-sm text-[var(--color-text-secondary)]">Current & predicted</p>
        </div>
      </div>

      <div className="space-y-4">
        {symptoms.map((symptom, index) => (
          <div key={index} className="glass-hover rounded-[var(--radius)] p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="font-semibold text-[var(--color-text)]">{symptom.name}</span>
              <span
                className={`px-3 py-1 rounded-full text-xs font-medium text-white bg-gradient-to-r ${getRiskColor(symptom.risk)}`}
              >
                {symptom.risk} risk
              </span>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-[var(--color-text-secondary)] mb-1">Current</p>
                <p className="text-2xl font-bold text-[var(--color-text)]">{symptom.current}/10</p>
              </div>
              <div>
                <p className="text-xs text-[var(--color-text-secondary)] mb-1">Predicted Tomorrow</p>
                <div className="flex items-center gap-2">
                  <p className="text-2xl font-bold text-[var(--color-text)]">{symptom.predicted}/10</p>
                  {symptom.predicted < symptom.current && <TrendingDown className="w-4 h-4 text-green-500" />}
                  {symptom.predicted > symptom.current && <TrendingUp className="w-4 h-4 text-red-500" />}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
