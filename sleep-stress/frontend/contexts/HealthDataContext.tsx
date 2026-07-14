"use client"

import { createContext, useContext, useState, useEffect, type ReactNode } from "react"
import type { SleepEntry, MoodEntry, SymptomEntry, ProfileData, GamificationData, HealthGoal } from "@/lib/data-export"
import {
  getSleepData,
  saveSleepEntry,
  getMoodData,
  saveMoodEntry,
  getSymptomsData,
  saveSymptomEntry,
  getProfileData,
  saveProfileData,
  getGamificationData,
  saveGamificationData,
  getHealthGoals,
  saveHealthGoals,
  updateGamificationOnEntry,
} from "@/lib/storage"

interface HealthDataContextType {
  sleepData: SleepEntry[]
  moodData: MoodEntry[]
  symptomsData: SymptomEntry[]
  profileData: ProfileData
  gamificationData: GamificationData
  healthGoals: HealthGoal[]
  addSleepEntry: (entry: SleepEntry) => void
  addMoodEntry: (entry: MoodEntry) => void
  addSymptomEntry: (entry: SymptomEntry) => void
  updateProfile: (profile: ProfileData) => void
  updateGamification: (data: GamificationData) => void
  updateGoals: (goals: HealthGoal[]) => void
  refreshData: () => void
}

const HealthDataContext = createContext<HealthDataContextType | undefined>(undefined)

export function HealthDataProvider({ children }: { children: ReactNode }) {
  const [sleepData, setSleepData] = useState<SleepEntry[]>([])
  const [moodData, setMoodData] = useState<MoodEntry[]>([])
  const [symptomsData, setSymptomsData] = useState<SymptomEntry[]>([])
  const [profileData, setProfileData] = useState<ProfileData>({
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
  const [gamificationData, setGamificationData] = useState<GamificationData>({
    level: 1,
    xp: 0,
    xpToNextLevel: 500,
    currentStreak: 0,
    longestStreak: 0,
    badges: [],
    achievements: [],
  })
  const [healthGoals, setHealthGoals] = useState<HealthGoal[]>([])

  // Load data from localStorage on mount
  useEffect(() => {
    refreshData()
  }, [])

  const refreshData = () => {
    setSleepData(getSleepData())
    setMoodData(getMoodData())
    setSymptomsData(getSymptomsData())
    setProfileData(getProfileData())
    setGamificationData(getGamificationData())
    setHealthGoals(getHealthGoals())
  }

  const addSleepEntry = (entry: SleepEntry) => {
    saveSleepEntry(entry)
    updateGamificationOnEntry()
    refreshData()
  }

  const addMoodEntry = (entry: MoodEntry) => {
    saveMoodEntry(entry)
    updateGamificationOnEntry()
    refreshData()
  }

  const addSymptomEntry = (entry: SymptomEntry) => {
    saveSymptomEntry(entry)
    updateGamificationOnEntry()
    refreshData()
  }

  const updateProfile = (profile: ProfileData) => {
    saveProfileData(profile)
    refreshData()
  }

  const updateGamification = (data: GamificationData) => {
    saveGamificationData(data)
    refreshData()
  }

  const updateGoals = (goals: HealthGoal[]) => {
    saveHealthGoals(goals)
    refreshData()
  }

  return (
    <HealthDataContext.Provider
      value={{
        sleepData,
        moodData,
        symptomsData,
        profileData,
        gamificationData,
        healthGoals,
        addSleepEntry,
        addMoodEntry,
        addSymptomEntry,
        updateProfile,
        updateGamification,
        updateGoals,
        refreshData,
      }}
    >
      {children}
    </HealthDataContext.Provider>
  )
}

export function useHealthData() {
  const context = useContext(HealthDataContext)
  if (context === undefined) {
    throw new Error("useHealthData must be used within a HealthDataProvider")
  }
  return context
}
