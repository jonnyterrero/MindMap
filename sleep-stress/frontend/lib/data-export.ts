// Data type interfaces
export interface SleepEntry {
  date: string
  hours: number
  quality: number
}

export interface MoodEntry {
  date: string
  mood: number
  stress: number
}

export interface SymptomEntry {
  date: string
  gi: number
  skin: number
  migraine: number
}

export interface ProfileData {
  name: string
  email: string
  age: number
  gender: string
  height: number
  weight: number
  bmi: number
  bmiCategory: string
  conditions: string[]
  medications: string[]
  allergies: string[]
  activityLevel: string
}

export interface GamificationData {
  level: number
  xp: number
  xpToNextLevel: number
  currentStreak: number
  longestStreak: number
  badges: Array<{ name: string; earned: boolean; progress: number }>
  achievements: Array<{ name: string; description: string; date: string }>
}

export interface HealthGoal {
  id: number
  name: string
  progress: number
  target: number
}

// Convert data to CSV format
export function convertToCSV(data: any[], headers: string[]): string {
  const csvRows = []

  // Add headers
  csvRows.push(headers.join(","))

  // Add data rows
  for (const row of data) {
    const values = headers.map((header) => {
      const value = row[header.toLowerCase().replace(" ", "")]
      // Handle arrays and objects
      if (Array.isArray(value)) {
        return `"${value.join("; ")}"`
      }
      // Escape quotes and wrap in quotes if contains comma
      const escaped = String(value).replace(/"/g, '""')
      return escaped.includes(",") ? `"${escaped}"` : escaped
    })
    csvRows.push(values.join(","))
  }

  return csvRows.join("\n")
}

// Download file helper
export function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

// Export sleep data
export function exportSleepData(data: SleepEntry[], format: "csv" | "json") {
  const timestamp = new Date().toISOString().split("T")[0]

  if (format === "csv") {
    const csv = convertToCSV(data, ["Date", "Hours", "Quality"])
    downloadFile(csv, `sleep-data-${timestamp}.csv`, "text/csv")
  } else {
    const json = JSON.stringify(data, null, 2)
    downloadFile(json, `sleep-data-${timestamp}.json`, "application/json")
  }
}

// Export mood data
export function exportMoodData(data: MoodEntry[], format: "csv" | "json") {
  const timestamp = new Date().toISOString().split("T")[0]

  if (format === "csv") {
    const csv = convertToCSV(data, ["Date", "Mood", "Stress"])
    downloadFile(csv, `mood-data-${timestamp}.csv`, "text/csv")
  } else {
    const json = JSON.stringify(data, null, 2)
    downloadFile(json, `mood-data-${timestamp}.json`, "application/json")
  }
}

// Export symptoms data
export function exportSymptomsData(data: SymptomEntry[], format: "csv" | "json") {
  const timestamp = new Date().toISOString().split("T")[0]

  if (format === "csv") {
    const csv = convertToCSV(data, ["Date", "GI", "Skin", "Migraine"])
    downloadFile(csv, `symptoms-data-${timestamp}.csv`, "text/csv")
  } else {
    const json = JSON.stringify(data, null, 2)
    downloadFile(json, `symptoms-data-${timestamp}.json`, "application/json")
  }
}

// Export profile data
export function exportProfileData(data: ProfileData, format: "csv" | "json") {
  const timestamp = new Date().toISOString().split("T")[0]

  if (format === "csv") {
    // Convert profile to array format for CSV
    const profileArray = [
      {
        name: data.name,
        email: data.email,
        age: data.age,
        gender: data.gender,
        height: data.height,
        weight: data.weight,
        bmi: data.bmi,
        bmicategory: data.bmiCategory,
        conditions: data.conditions.join("; "),
        medications: data.medications.join("; "),
        allergies: data.allergies.join("; "),
        activitylevel: data.activityLevel,
      },
    ]
    const csv = convertToCSV(profileArray, [
      "Name",
      "Email",
      "Age",
      "Gender",
      "Height",
      "Weight",
      "BMI",
      "BMICategory",
      "Conditions",
      "Medications",
      "Allergies",
      "ActivityLevel",
    ])
    downloadFile(csv, `profile-data-${timestamp}.csv`, "text/csv")
  } else {
    const json = JSON.stringify(data, null, 2)
    downloadFile(json, `profile-data-${timestamp}.json`, "application/json")
  }
}

// Export all data
export function exportAllData(
  sleepData: SleepEntry[],
  moodData: MoodEntry[],
  symptomsData: SymptomEntry[],
  profileData: ProfileData,
  gamificationData: GamificationData,
  healthGoals: HealthGoal[],
  format: "json",
) {
  const timestamp = new Date().toISOString().split("T")[0]

  const allData = {
    exportDate: new Date().toISOString(),
    profile: profileData,
    sleep: sleepData,
    mood: moodData,
    symptoms: symptomsData,
    gamification: gamificationData,
    healthGoals: healthGoals,
  }

  const json = JSON.stringify(allData, null, 2)
  downloadFile(json, `health-data-complete-${timestamp}.json`, "application/json")
}
