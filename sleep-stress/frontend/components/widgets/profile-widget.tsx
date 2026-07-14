"use client"

import { Activity, Heart } from "lucide-react"
import { useHealthData } from "@/contexts/HealthDataContext"

export function ProfileWidget() {
  const { profileData } = useHealthData()

  const calculateCompleteness = () => {
    let filled = 0
    const total = 11

    if (profileData.name) filled++
    if (profileData.email) filled++
    if (profileData.age > 0) filled++
    if (profileData.gender) filled++
    if (profileData.height > 0) filled++
    if (profileData.weight > 0) filled++
    if (profileData.bmi > 0) filled++
    if (profileData.conditions.length > 0) filled++
    if (profileData.medications.length > 0) filled++
    if (profileData.allergies.length > 0) filled++
    if (profileData.activityLevel) filled++

    return Math.round((filled / total) * 100)
  }

  const completeness = calculateCompleteness()
  const displayName = profileData.name || "Guest User"
  const initial = displayName.charAt(0).toUpperCase()

  return (
    <div className="glass rounded-[var(--radius-lg)] p-6">
      <div className="flex items-center gap-4 mb-6">
        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-secondary)] flex items-center justify-center text-white text-2xl font-bold">
          {initial}
        </div>
        <div>
          <h3 className="text-xl font-bold text-[var(--color-text)]">{displayName}</h3>
          <p className="text-sm text-[var(--color-text-secondary)]">
            {profileData.age > 0 ? `${profileData.age} years old` : "Age not set"}
          </p>
        </div>
      </div>

      <div className="space-y-4 mb-6">
        <div className="flex items-center justify-between">
          <span className="text-sm text-[var(--color-text-secondary)]">Profile Completeness</span>
          <span className="text-sm font-bold text-[var(--color-text)]">{completeness}%</span>
        </div>
        <div className="h-2 bg-white/50 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] rounded-full transition-all duration-500"
            style={{ width: `${completeness}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-gradient-to-br from-[var(--color-glass-blue)] to-white/50 rounded-[var(--radius)] p-3">
          <div className="flex items-center gap-2 mb-1">
            <Activity className="w-4 h-4 text-[var(--color-primary)]" />
            <span className="text-xs text-[var(--color-text-secondary)]">BMI</span>
          </div>
          <p className="text-xl font-bold text-[var(--color-text)]">
            {profileData.bmi > 0 ? profileData.bmi.toFixed(1) : "--"}
          </p>
          <p className="text-xs text-[var(--color-text-muted)]">{profileData.bmiCategory || "Not calculated"}</p>
        </div>

        <div className="bg-gradient-to-br from-[var(--color-glass-pink)] to-white/50 rounded-[var(--radius)] p-3">
          <div className="flex items-center gap-2 mb-1">
            <Heart className="w-4 h-4 text-[var(--color-secondary-dark)]" />
            <span className="text-xs text-[var(--color-text-secondary)]">Conditions</span>
          </div>
          <p className="text-xl font-bold text-[var(--color-text)]">{profileData.conditions.length}</p>
          <p className="text-xs text-[var(--color-text-muted)]">tracked</p>
        </div>
      </div>

      {profileData.conditions.length > 0 ? (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase">Health Conditions</p>
          <div className="flex flex-wrap gap-2">
            {profileData.conditions.map((condition, index) => (
              <span
                key={index}
                className="px-3 py-1 bg-white/60 rounded-full text-xs font-medium text-[var(--color-text)]"
              >
                {condition}
              </span>
            ))}
          </div>
        </div>
      ) : (
        <div className="text-center py-4">
          <p className="text-sm text-[var(--color-text-secondary)]">No conditions added yet</p>
          <p className="text-xs text-[var(--color-text-muted)]">Update your profile to add conditions</p>
        </div>
      )}
    </div>
  )
}
