// LocalStorage utility functions for health tracking data

import type { SleepEntry, MoodEntry, SymptomEntry, ProfileData, GamificationData, HealthGoal } from "./data-export"

// Storage keys
const STORAGE_KEYS = {
  SLEEP: "health_tracker_sleep",
  MOOD: "health_tracker_mood",
  SYMPTOMS: "health_tracker_symptoms",
  PROFILE: "health_tracker_profile",
  GAMIFICATION: "health_tracker_gamification",
  GOALS: "health_tracker_goals",
} as const

// Generic storage functions
function getFromStorage<T>(key: string, defaultValue: T): T {
  if (typeof window === "undefined") return defaultValue
  try {
    const item = window.localStorage.getItem(key)
    return item ? JSON.parse(item) : defaultValue
  } catch (error) {
    console.error(`Error reading from localStorage key "${key}":`, error)
    return defaultValue
  }
}

function saveToStorage<T>(key: string, value: T): void {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(key, JSON.stringify(value))
  } catch (error) {
    console.error(`Error saving to localStorage key "${key}":`, error)
  }
}

// Sleep data functions
export function getSleepData(): SleepEntry[] {
  return getFromStorage<SleepEntry[]>(STORAGE_KEYS.SLEEP, [])
}

export function saveSleepEntry(entry: SleepEntry): void {
  const data = getSleepData()
  // Check if entry for this date already exists
  const existingIndex = data.findIndex((e) => e.date === entry.date)
  if (existingIndex >= 0) {
    data[existingIndex] = entry
  } else {
    data.push(entry)
  }
  // Sort by date descending
  data.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
  saveToStorage(STORAGE_KEYS.SLEEP, data)
}

export function deleteSleepEntry(date: string): void {
  const data = getSleepData().filter((e) => e.date !== date)
  saveToStorage(STORAGE_KEYS.SLEEP, data)
}

// Mood data functions
export function getMoodData(): MoodEntry[] {
  return getFromStorage<MoodEntry[]>(STORAGE_KEYS.MOOD, [])
}

export function saveMoodEntry(entry: MoodEntry): void {
  const data = getMoodData()
  // Check if entry for this date already exists
  const existingIndex = data.findIndex((e) => e.date === entry.date)
  if (existingIndex >= 0) {
    data[existingIndex] = entry
  } else {
    data.push(entry)
  }
  // Sort by date descending
  data.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
  saveToStorage(STORAGE_KEYS.MOOD, data)
}

export function deleteMoodEntry(date: string): void {
  const data = getMoodData().filter((e) => e.date !== date)
  saveToStorage(STORAGE_KEYS.MOOD, data)
}

// Symptoms data functions
export function getSymptomsData(): SymptomEntry[] {
  return getFromStorage<SymptomEntry[]>(STORAGE_KEYS.SYMPTOMS, [])
}

export function saveSymptomEntry(entry: SymptomEntry): void {
  const data = getSymptomsData()
  // Check if entry for this date already exists
  const existingIndex = data.findIndex((e) => e.date === entry.date)
  if (existingIndex >= 0) {
    data[existingIndex] = entry
  } else {
    data.push(entry)
  }
  // Sort by date descending
  data.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
  saveToStorage(STORAGE_KEYS.SYMPTOMS, data)
}

export function deleteSymptomEntry(date: string): void {
  const data = getSymptomsData().filter((e) => e.date !== date)
  saveToStorage(STORAGE_KEYS.SYMPTOMS, data)
}

// Profile data functions
export function getProfileData(): ProfileData {
  return getFromStorage<ProfileData>(STORAGE_KEYS.PROFILE, {
    name: "",
    email: "",
    age: 0,
    gender: "",
    height: 0,
    weight: 0,
    bmi: 0,
    bmiCategory: "",
    conditions: [],
    medications: [],
    allergies: [],
    activityLevel: "",
  })
}

export function saveProfileData(profile: ProfileData): void {
  saveToStorage(STORAGE_KEYS.PROFILE, profile)
}

// Gamification data functions
export function getGamificationData(): GamificationData {
  return getFromStorage<GamificationData>(STORAGE_KEYS.GAMIFICATION, {
    level: 1,
    xp: 0,
    xpToNextLevel: 500,
    currentStreak: 0,
    longestStreak: 0,
    badges: [
      { name: "Early Bird", earned: false, progress: 0 },
      { name: "Consistency King", earned: false, progress: 0 },
      { name: "Mood Master", earned: false, progress: 0 },
      { name: "Sleep Champion", earned: false, progress: 0 },
      { name: "Wellness Warrior", earned: false, progress: 0 },
    ],
    achievements: [],
  })
}

export function saveGamificationData(data: GamificationData): void {
  saveToStorage(STORAGE_KEYS.GAMIFICATION, data)
}

// Update gamification based on new entry
export function updateGamificationOnEntry(): void {
  const gamification = getGamificationData()

  // Add XP for logging
  gamification.xp += 50

  // Check for level up
  while (gamification.xp >= gamification.xpToNextLevel) {
    gamification.level += 1
    gamification.xp -= gamification.xpToNextLevel
    gamification.xpToNextLevel = Math.floor(gamification.xpToNextLevel * 1.5)
  }

  // Update streak
  const today = new Date().toISOString().split("T")[0]
  const sleepData = getSleepData()
  const moodData = getMoodData()
  const symptomsData = getSymptomsData()

  // Check if user logged today
  const loggedToday =
    sleepData.some((e) => e.date === today) ||
    moodData.some((e) => e.date === today) ||
    symptomsData.some((e) => e.date === today)

  if (loggedToday) {
    gamification.currentStreak += 1
    if (gamification.currentStreak > gamification.longestStreak) {
      gamification.longestStreak = gamification.currentStreak
    }
  }

  saveGamificationData(gamification)
}

// Health goals functions
export function getHealthGoals(): HealthGoal[] {
  return getFromStorage<HealthGoal[]>(STORAGE_KEYS.GOALS, [
    { id: 1, name: "Sleep 7.5 hours daily", progress: 0, target: 100 },
    { id: 2, name: "Reduce stress below 5", progress: 0, target: 100 },
    { id: 3, name: "Maintain mood above 7", progress: 0, target: 100 },
  ])
}

export function saveHealthGoals(goals: HealthGoal[]): void {
  saveToStorage(STORAGE_KEYS.GOALS, goals)
}

export function updateHealthGoal(id: number, progress: number): void {
  const goals = getHealthGoals()
  const goal = goals.find((g) => g.id === id)
  if (goal) {
    goal.progress = Math.min(progress, goal.target)
    saveHealthGoals(goals)
  }
}

// Calculate statistics
export function calculateSleepStats(days = 7) {
  const data = getSleepData().slice(0, days)
  if (data.length === 0) return { avgHours: 0, avgQuality: 0, count: 0 }

  const avgHours = data.reduce((sum, e) => sum + e.hours, 0) / data.length
  const avgQuality = data.reduce((sum, e) => sum + e.quality, 0) / data.length

  return { avgHours, avgQuality, count: data.length }
}

export function calculateMoodStats(days = 7) {
  const data = getMoodData().slice(0, days)
  if (data.length === 0) return { avgMood: 0, avgStress: 0, count: 0 }

  const avgMood = data.reduce((sum, e) => sum + e.mood, 0) / data.length
  const avgStress = data.reduce((sum, e) => sum + e.stress, 0) / data.length

  return { avgMood, avgStress, count: data.length }
}

export function calculateSymptomStats(days = 7) {
  const data = getSymptomsData().slice(0, days)
  if (data.length === 0) return { avgGi: 0, avgSkin: 0, avgMigraine: 0, count: 0 }

  const avgGi = data.reduce((sum, e) => sum + e.gi, 0) / data.length
  const avgSkin = data.reduce((sum, e) => sum + e.skin, 0) / data.length
  const avgMigraine = data.reduce((sum, e) => sum + e.migraine, 0) / data.length

  return { avgGi, avgSkin, avgMigraine, count: data.length }
}
