"use client"

import { Activity, Heart, Brain, Settings, Edit, Download } from "lucide-react"
import { useState } from "react"
import { ExportDataDialog } from "../export-data-dialog"
import { EditProfileForm } from "../forms/edit-profile-form"
import { useHealthData } from "@/contexts/HealthDataContext"

export function ProfilePage() {
  const [showExportDialog, setShowExportDialog] = useState(false)
  const [showEditForm, setShowEditForm] = useState(false)
  const { profileData, healthGoals } = useHealthData()

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
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="glass rounded-[var(--radius-lg)] p-6 md:p-8">
        <div className="flex items-center gap-4">
          <div className="w-20 h-20 rounded-full bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-secondary)] flex items-center justify-center text-white text-3xl font-bold">
            {initial}
          </div>
          <div className="flex-1">
            <h1 className="text-3xl md:text-4xl font-bold gradient-text">{displayName}</h1>
            <p className="text-[var(--color-text-secondary)]">{profileData.email || "No email set"}</p>
          </div>
          <button
            onClick={() => setShowEditForm(true)}
            className="glass-hover px-4 py-2 rounded-[var(--radius)] flex items-center gap-2 text-[var(--color-text)]"
          >
            <Edit className="w-4 h-4" />
            <span className="hidden md:inline">Edit Profile</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column */}
        <div className="lg:col-span-2 space-y-6">
          {/* Basic Info */}
          <div className="glass rounded-[var(--radius-lg)] p-6">
            <h2 className="text-xl font-bold text-[var(--color-text)] mb-6">Basic Information</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-xs text-[var(--color-text-secondary)] mb-1">Age</p>
                <p className="text-lg font-bold text-[var(--color-text)]">
                  {profileData.age > 0 ? `${profileData.age} years` : "Not set"}
                </p>
              </div>
              <div>
                <p className="text-xs text-[var(--color-text-secondary)] mb-1">Gender</p>
                <p className="text-lg font-bold text-[var(--color-text)]">{profileData.gender || "Not set"}</p>
              </div>
              <div>
                <p className="text-xs text-[var(--color-text-secondary)] mb-1">Height</p>
                <p className="text-lg font-bold text-[var(--color-text)]">
                  {profileData.height > 0 ? `${profileData.height} cm` : "Not set"}
                </p>
              </div>
              <div>
                <p className="text-xs text-[var(--color-text-secondary)] mb-1">Weight</p>
                <p className="text-lg font-bold text-[var(--color-text)]">
                  {profileData.weight > 0 ? `${profileData.weight} kg` : "Not set"}
                </p>
              </div>
            </div>

            <div className="mt-6 pt-6 border-t border-[var(--color-border)]">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <p className="text-sm font-semibold text-[var(--color-text)]">
                    BMI: {profileData.bmi > 0 ? profileData.bmi.toFixed(1) : "Not calculated"}
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)]">{profileData.bmiCategory || "N/A"}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-semibold text-[var(--color-text)]">Activity Level</p>
                  <p className="text-xs text-[var(--color-text-secondary)]">{profileData.activityLevel || "Not set"}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Health Conditions */}
          <div className="glass rounded-[var(--radius-lg)] p-6">
            <h2 className="text-xl font-bold text-[var(--color-text)] mb-6">Health Information</h2>

            <div className="space-y-6">
              <div>
                <p className="text-sm font-semibold text-[var(--color-text)] mb-3">Conditions</p>
                {profileData.conditions.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {profileData.conditions.map((condition, index) => (
                      <span
                        key={index}
                        className="px-4 py-2 bg-gradient-to-r from-[var(--color-glass-purple)] to-[var(--color-glass-blue)] rounded-full text-sm font-medium text-[var(--color-text)]"
                      >
                        {condition}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-[var(--color-text-secondary)]">No conditions added</p>
                )}
              </div>

              <div>
                <p className="text-sm font-semibold text-[var(--color-text)] mb-3">Medications</p>
                {profileData.medications.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {profileData.medications.map((medication, index) => (
                      <span
                        key={index}
                        className="px-4 py-2 bg-gradient-to-r from-[var(--color-glass-pink)] to-[var(--color-glass-purple)] rounded-full text-sm font-medium text-[var(--color-text)]"
                      >
                        {medication}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-[var(--color-text-secondary)]">No medications added</p>
                )}
              </div>

              <div>
                <p className="text-sm font-semibold text-[var(--color-text)] mb-3">Allergies</p>
                {profileData.allergies.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {profileData.allergies.map((allergy, index) => (
                      <span
                        key={index}
                        className="px-4 py-2 bg-gradient-to-r from-orange-100 to-red-100 rounded-full text-sm font-medium text-red-700"
                      >
                        {allergy}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-[var(--color-text-secondary)]">No allergies added</p>
                )}
              </div>
            </div>
          </div>

          {/* Health Goals */}
          <div className="glass rounded-[var(--radius-lg)] p-6">
            <h2 className="text-xl font-bold text-[var(--color-text)] mb-6">Health Goals</h2>
            {healthGoals.length > 0 ? (
              <div className="space-y-4">
                {healthGoals.map((goal) => (
                  <div key={goal.id} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium text-[var(--color-text)]">{goal.name}</p>
                      <span className="text-sm font-bold text-[var(--color-text)]">{goal.progress}%</span>
                    </div>
                    <div className="h-3 bg-white/50 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] rounded-full transition-all duration-500"
                        style={{ width: `${goal.progress}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-center py-8 text-[var(--color-text-secondary)]">No health goals set yet</p>
            )}
          </div>
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          {/* Profile Completeness */}
          <div className="glass rounded-[var(--radius-lg)] p-6">
            <h3 className="text-lg font-bold text-[var(--color-text)] mb-4">Profile Completeness</h3>
            <div className="flex items-center justify-center mb-4">
              <div className="relative w-32 h-32">
                <svg className="w-32 h-32 transform -rotate-90">
                  <circle
                    cx="64"
                    cy="64"
                    r="56"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="none"
                    className="text-white/50"
                  />
                  <circle
                    cx="64"
                    cy="64"
                    r="56"
                    stroke="url(#gradient)"
                    strokeWidth="8"
                    fill="none"
                    strokeDasharray={`${(completeness / 100) * 351.86} 351.86`}
                    className="transition-all duration-500"
                  />
                  <defs>
                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="var(--color-primary)" />
                      <stop offset="100%" stopColor="var(--color-secondary)" />
                    </linearGradient>
                  </defs>
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-3xl font-bold text-[var(--color-text)]">{completeness}%</span>
                </div>
              </div>
            </div>
            <p className="text-sm text-[var(--color-text-secondary)] text-center">
              Complete your profile to get better insights
            </p>
          </div>

          {/* Quick Stats */}
          <div className="glass rounded-[var(--radius-lg)] p-6">
            <h3 className="text-lg font-bold text-[var(--color-text)] mb-4">Quick Stats</h3>
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-primary-dark)] flex items-center justify-center">
                  <Activity className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-xs text-[var(--color-text-secondary)]">Conditions Tracked</p>
                  <p className="text-lg font-bold text-[var(--color-text)]">{profileData.conditions.length}</p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-secondary)] to-[var(--color-secondary-dark)] flex items-center justify-center">
                  <Heart className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-xs text-[var(--color-text-secondary)]">Medications</p>
                  <p className="text-lg font-bold text-[var(--color-text)]">{profileData.medications.length}</p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent-dark)] flex items-center justify-center">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-xs text-[var(--color-text-secondary)]">Active Goals</p>
                  <p className="text-lg font-bold text-[var(--color-text)]">{healthGoals.length}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Settings */}
          <div className="glass rounded-[var(--radius-lg)] p-6">
            <h3 className="text-lg font-bold text-[var(--color-text)] mb-4">Settings</h3>
            <div className="space-y-3">
              <button className="w-full glass-hover rounded-[var(--radius)] p-3 flex items-center gap-3 text-left">
                <Settings className="w-5 h-5 text-[var(--color-text-secondary)]" />
                <span className="text-sm font-medium text-[var(--color-text)]">Account Settings</span>
              </button>
              <button className="w-full glass-hover rounded-[var(--radius)] p-3 flex items-center gap-3 text-left">
                <Settings className="w-5 h-5 text-[var(--color-text-secondary)]" />
                <span className="text-sm font-medium text-[var(--color-text)]">Privacy Settings</span>
              </button>
              <button className="w-full glass-hover rounded-[var(--radius)] p-3 flex items-center gap-3 text-left">
                <Settings className="w-5 h-5 text-[var(--color-text-secondary)]" />
                <span className="text-sm font-medium text-[var(--color-text)]">Notification Settings</span>
              </button>
              <button
                onClick={() => setShowExportDialog(true)}
                className="w-full glass-hover rounded-[var(--radius)] p-3 flex items-center gap-3 text-left bg-gradient-to-r from-[var(--color-glass-blue)] to-[var(--color-glass-purple)]"
              >
                <Download className="w-5 h-5 text-[var(--color-primary)]" />
                <span className="text-sm font-medium text-[var(--color-text)]">Export Data</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {showEditForm && <EditProfileForm onClose={() => setShowEditForm(false)} />}
      <ExportDataDialog isOpen={showExportDialog} onClose={() => setShowExportDialog(false)} />
    </div>
  )
}
