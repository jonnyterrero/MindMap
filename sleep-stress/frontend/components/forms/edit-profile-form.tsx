"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { User, X, Mail, Calendar, Ruler, Weight, Activity } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useHealthData } from "@/contexts/HealthDataContext"
import { toast } from "sonner"
import type { ProfileData } from "@/lib/data-export"

interface EditProfileFormProps {
  onClose: () => void
}

export function EditProfileForm({ onClose }: EditProfileFormProps) {
  const { profileData, updateProfile } = useHealthData()

  const [formData, setFormData] = useState<ProfileData>(profileData)

  useEffect(() => {
    setFormData(profileData)
  }, [profileData])

  const calculateBMI = (weight: number, height: number) => {
    if (weight > 0 && height > 0) {
      const heightInMeters = height / 100
      const bmi = weight / (heightInMeters * heightInMeters)
      return Math.round(bmi * 10) / 10
    }
    return 0
  }

  const getBMICategory = (bmi: number) => {
    if (bmi < 18.5) return "Underweight"
    if (bmi < 25) return "Normal weight"
    if (bmi < 30) return "Overweight"
    return "Obese"
  }

  const handleChange = (field: keyof ProfileData, value: any) => {
    const updated = { ...formData, [field]: value }

    if (field === "weight" || field === "height") {
      const bmi = calculateBMI(field === "weight" ? value : updated.weight, field === "height" ? value : updated.height)
      updated.bmi = bmi
      updated.bmiCategory = getBMICategory(bmi)
    }

    setFormData(updated)
  }

  const handleArrayChange = (field: "conditions" | "medications" | "allergies", value: string) => {
    const items = value
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0)
    setFormData({ ...formData, [field]: items })
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    updateProfile(formData)
    toast.success("Profile updated successfully!")
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50 overflow-y-auto">
      <div className="glass rounded-[var(--radius-lg)] p-6 w-full max-w-2xl my-8">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-secondary)] flex items-center justify-center">
              <User className="w-6 h-6 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-[var(--color-text)]">Edit Profile</h2>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-[var(--radius-sm)] hover:bg-white/50 flex items-center justify-center transition-colors"
          >
            <X className="w-5 h-5 text-[var(--color-text)]" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Basic Information */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-[var(--color-text)]">Basic Information</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-[var(--color-text)] flex items-center gap-2">
                  <User className="w-4 h-4" />
                  Full Name
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => handleChange("name", e.target.value)}
                  placeholder="Enter your name"
                  className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-[var(--color-text)] flex items-center gap-2">
                  <Mail className="w-4 h-4" />
                  Email
                </label>
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => handleChange("email", e.target.value)}
                  placeholder="your.email@example.com"
                  className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-[var(--color-text)] flex items-center gap-2">
                  <Calendar className="w-4 h-4" />
                  Age
                </label>
                <input
                  type="number"
                  value={formData.age || ""}
                  onChange={(e) => handleChange("age", Number(e.target.value))}
                  placeholder="25"
                  min="0"
                  max="120"
                  className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-[var(--color-text)]">Gender</label>
                <select
                  value={formData.gender}
                  onChange={(e) => handleChange("gender", e.target.value)}
                  className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                >
                  <option value="">Select gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Non-binary">Non-binary</option>
                  <option value="Prefer not to say">Prefer not to say</option>
                </select>
              </div>
            </div>
          </div>

          {/* Physical Measurements */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-[var(--color-text)]">Physical Measurements</h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-[var(--color-text)] flex items-center gap-2">
                  <Ruler className="w-4 h-4" />
                  Height (cm)
                </label>
                <input
                  type="number"
                  value={formData.height || ""}
                  onChange={(e) => handleChange("height", Number(e.target.value))}
                  placeholder="170"
                  min="0"
                  max="300"
                  className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-[var(--color-text)] flex items-center gap-2">
                  <Weight className="w-4 h-4" />
                  Weight (kg)
                </label>
                <input
                  type="number"
                  value={formData.weight || ""}
                  onChange={(e) => handleChange("weight", Number(e.target.value))}
                  placeholder="70"
                  min="0"
                  max="500"
                  step="0.1"
                  className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-[var(--color-text)]">BMI</label>
                <div className="px-4 py-3 bg-white/40 border border-[var(--color-border)] rounded-[var(--radius)]">
                  <p className="text-lg font-bold text-[var(--color-text)]">
                    {formData.bmi > 0 ? formData.bmi.toFixed(1) : "--"}
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)]">
                    {formData.bmiCategory || "Not calculated"}
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-[var(--color-text)] flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Activity Level
              </label>
              <select
                value={formData.activityLevel}
                onChange={(e) => handleChange("activityLevel", e.target.value)}
                className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              >
                <option value="">Select activity level</option>
                <option value="Sedentary">Sedentary (little or no exercise)</option>
                <option value="Light">Light (exercise 1-3 days/week)</option>
                <option value="Moderate">Moderate (exercise 3-5 days/week)</option>
                <option value="Active">Active (exercise 6-7 days/week)</option>
                <option value="Very Active">Very Active (intense exercise daily)</option>
              </select>
            </div>
          </div>

          {/* Health Information */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-[var(--color-text)]">Health Information</h3>

            <div className="space-y-2">
              <label className="text-sm font-medium text-[var(--color-text)]">Health Conditions</label>
              <input
                type="text"
                value={formData.conditions.join(", ")}
                onChange={(e) => handleArrayChange("conditions", e.target.value)}
                placeholder="Anxiety, IBS, Migraine (comma separated)"
                className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              />
              <p className="text-xs text-[var(--color-text-muted)]">Separate multiple conditions with commas</p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-[var(--color-text)]">Medications</label>
              <input
                type="text"
                value={formData.medications.join(", ")}
                onChange={(e) => handleArrayChange("medications", e.target.value)}
                placeholder="Medication names (comma separated)"
                className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-[var(--color-text)]">Allergies</label>
              <input
                type="text"
                value={formData.allergies.join(", ")}
                onChange={(e) => handleArrayChange("allergies", e.target.value)}
                placeholder="Allergy names (comma separated)"
                className="w-full px-4 py-3 bg-white/60 border border-[var(--color-border)] rounded-[var(--radius)] text-[var(--color-text)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              />
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex gap-3">
            <Button
              type="button"
              onClick={onClose}
              className="flex-1 py-3 bg-white/60 text-[var(--color-text)] font-semibold rounded-[var(--radius)] hover:bg-white/80 transition-colors"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              className="flex-1 py-3 bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] text-white font-semibold rounded-[var(--radius)] hover:opacity-90 transition-opacity"
            >
              Save Changes
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
