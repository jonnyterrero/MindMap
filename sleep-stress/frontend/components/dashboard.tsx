"use client"

import { Moon, Heart, Brain, Award, Calendar } from "lucide-react"
import { SleepWidget } from "./widgets/sleep-widget"
import { MoodWidget } from "./widgets/mood-widget"
import { InsightsWidget } from "./widgets/insights-widget"
import { StatsWidget } from "./widgets/stats-widget"
import { GamificationWidget } from "./widgets/gamification-widget"
import { QuickLogWidget } from "./widgets/quick-log-widget"
import { CorrelationWidget } from "./widgets/correlation-widget"
import { SymptomsWidget } from "./widgets/symptoms-widget"
import { CopingStrategiesWidget } from "./widgets/coping-strategies-widget"
import { ProfileWidget } from "./widgets/profile-widget"
import { CrossAppWidget } from "./widgets/cross-app-widget"
import { useHealthData } from "@/contexts/HealthDataContext"
import { calculateSleepStats, calculateMoodStats } from "@/lib/storage"

export function Dashboard() {
  const { profileData, gamificationData, sleepData, moodData } = useHealthData()

  const sleepStats = calculateSleepStats(7)
  const moodStats = calculateMoodStats(7)

  const avgSleepQuality = sleepStats.count > 0 ? sleepStats.avgQuality.toFixed(1) : "0.0"
  const avgMood = moodStats.count > 0 ? moodStats.avgMood.toFixed(1) : "0.0"
  const avgStress = moodStats.count > 0 ? moodStats.avgStress.toFixed(1) : "0.0"

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <header className="glass rounded-[var(--radius-lg)] p-6 md:p-8">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold gradient-text mb-2">
              Welcome back, {profileData.name || "there"}
            </h1>
            <p className="text-[var(--color-text-secondary)]">Here's your health overview for today</p>
          </div>
          <div className="flex items-center gap-3">
            <button className="glass glass-hover px-6 py-3 rounded-[var(--radius)] font-medium text-[var(--color-text)] flex items-center gap-2">
              <Calendar className="w-5 h-5" />
              Today
            </button>
          </div>
        </div>
      </header>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatsWidget
          icon={<Moon className="w-6 h-6" />}
          label="Sleep Quality"
          value={avgSleepQuality}
          unit="/10"
          trend="+0.5"
          color="primary"
        />
        <StatsWidget
          icon={<Heart className="w-6 h-6" />}
          label="Mood Score"
          value={avgMood}
          unit="/10"
          trend="+1.2"
          color="secondary"
        />
        <StatsWidget
          icon={<Brain className="w-6 h-6" />}
          label="Stress Level"
          value={avgStress}
          unit="/10"
          trend="-0.8"
          color="accent"
        />
        <StatsWidget
          icon={<Award className="w-6 h-6" />}
          label="Current Streak"
          value={gamificationData.currentStreak.toString()}
          unit="days"
          trend="+1"
          color="success"
        />
      </div>

      {/* Main Widgets Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Large Widgets */}
        <div className="lg:col-span-2 space-y-6">
          <SleepWidget />
          <MoodWidget />
          <CorrelationWidget />
          <SymptomsWidget />
        </div>

        {/* Right Column - Sidebar Widgets */}
        <div className="space-y-6">
          <QuickLogWidget />
          <GamificationWidget />
          <InsightsWidget />
          <ProfileWidget />
          <CopingStrategiesWidget />
          <CrossAppWidget />
        </div>
      </div>
    </div>
  )
}
