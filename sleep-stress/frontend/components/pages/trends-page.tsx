"use client"

import { TrendingUp, Calendar, Moon, Heart, Brain, Activity } from "lucide-react"
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts"
import { useHealthData } from "@/contexts/HealthDataContext"
import { useState } from "react"

export function TrendsPage() {
  const { sleepData, moodData, symptomsData } = useHealthData()
  const [timeRange, setTimeRange] = useState(30)

  const getChartData = (days: number) => {
    const sleep = sleepData
      .slice(0, days)
      .reverse()
      .map((entry) => ({
        date: new Date(entry.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
        hours: entry.hours,
        quality: entry.quality,
      }))

    const mood = moodData
      .slice(0, days)
      .reverse()
      .map((entry) => ({
        date: new Date(entry.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
        mood: entry.mood,
        stress: entry.stress,
      }))

    const symptoms = symptomsData
      .slice(0, days)
      .reverse()
      .map((entry) => ({
        date: new Date(entry.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
        gi: entry.gi,
        skin: entry.skin,
        migraine: entry.migraine,
      }))

    return { sleep, mood, symptoms }
  }

  const chartData = getChartData(timeRange)

  const getWeeklyAverages = () => {
    const weeks = []
    const weeksToShow = Math.min(4, Math.ceil(sleepData.length / 7))

    for (let i = 0; i < weeksToShow; i++) {
      const start = i * 7
      const end = start + 7
      const weekSleep = sleepData.slice(start, end)
      const weekMood = moodData.slice(start, end)

      if (weekSleep.length > 0 || weekMood.length > 0) {
        weeks.push({
          week: `Week ${weeksToShow - i}`,
          sleep: weekSleep.length > 0 ? weekSleep.reduce((sum, e) => sum + e.hours, 0) / weekSleep.length : 0,
          mood: weekMood.length > 0 ? weekMood.reduce((sum, e) => sum + e.mood, 0) / weekMood.length : 0,
          stress: weekMood.length > 0 ? weekMood.reduce((sum, e) => sum + e.stress, 0) / weekMood.length : 0,
        })
      }
    }

    return weeks.reverse()
  }

  const weeklyAverages = getWeeklyAverages()

  const calculateStats = () => {
    const recentData = {
      sleep: sleepData.slice(0, timeRange),
      mood: moodData.slice(0, timeRange),
      symptoms: symptomsData.slice(0, timeRange),
    }

    const avgSleep =
      recentData.sleep.length > 0
        ? (recentData.sleep.reduce((sum, e) => sum + e.hours, 0) / recentData.sleep.length).toFixed(1)
        : "0.0"

    const avgQuality =
      recentData.sleep.length > 0
        ? (recentData.sleep.reduce((sum, e) => sum + e.quality, 0) / recentData.sleep.length).toFixed(1)
        : "0.0"

    const avgMood =
      recentData.mood.length > 0
        ? (recentData.mood.reduce((sum, e) => sum + e.mood, 0) / recentData.mood.length).toFixed(1)
        : "0.0"

    const avgStress =
      recentData.mood.length > 0
        ? (recentData.mood.reduce((sum, e) => sum + e.stress, 0) / recentData.mood.length).toFixed(1)
        : "0.0"

    // Find best days
    const bestSleep =
      recentData.sleep.length > 0
        ? recentData.sleep.reduce((best, curr) => (curr.hours > best.hours ? curr : best))
        : null

    const bestMood =
      recentData.mood.length > 0 ? recentData.mood.reduce((best, curr) => (curr.mood > best.mood ? curr : best)) : null

    const lowestStress =
      recentData.mood.length > 0
        ? recentData.mood.reduce((best, curr) => (curr.stress < best.stress ? curr : best))
        : null

    const lowestSymptoms =
      recentData.symptoms.length > 0
        ? recentData.symptoms.reduce((best, curr) => {
            const currTotal = curr.gi + curr.skin + curr.migraine
            const bestTotal = best.gi + best.skin + best.migraine
            return currTotal < bestTotal ? curr : best
          })
        : null

    return {
      avgSleep,
      avgQuality,
      avgMood,
      avgStress,
      bestSleep: bestSleep
        ? new Date(bestSleep.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })
        : "N/A",
      bestMood: bestMood
        ? new Date(bestMood.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })
        : "N/A",
      lowestStress: lowestStress
        ? new Date(lowestStress.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })
        : "N/A",
      lowestSymptoms: lowestSymptoms
        ? new Date(lowestSymptoms.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })
        : "N/A",
    }
  }

  const stats = calculateStats()

  const hasData = sleepData.length > 0 || moodData.length > 0 || symptomsData.length > 0

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="glass rounded-[var(--radius-lg)] p-6 md:p-8">
        <div className="flex items-center gap-4 mb-2">
          <div className="w-14 h-14 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-accent)] flex items-center justify-center">
            <TrendingUp className="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 className="text-3xl md:text-4xl font-bold gradient-text">Health Trends</h1>
            <p className="text-[var(--color-text-secondary)]">Visualize your health data over time</p>
          </div>
        </div>
      </div>

      {!hasData ? (
        <div className="glass rounded-[var(--radius-lg)] p-12 text-center">
          <TrendingUp className="w-16 h-16 text-[var(--color-text-secondary)] mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-[var(--color-text)] mb-2">No Data Yet</h2>
          <p className="text-[var(--color-text-secondary)] mb-6">
            Start logging your sleep, mood, and symptoms to see trends and insights here
          </p>
        </div>
      ) : (
        <>
          {/* Time Range Selector */}
          <div className="glass rounded-[var(--radius-lg)] p-4">
            <div className="flex items-center gap-3 overflow-x-auto">
              <Calendar className="w-5 h-5 text-[var(--color-text-secondary)] flex-shrink-0" />
              <button
                onClick={() => setTimeRange(7)}
                className={`px-4 py-2 rounded-[var(--radius)] font-medium whitespace-nowrap ${
                  timeRange === 7
                    ? "bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] text-white"
                    : "glass-hover text-[var(--color-text)]"
                }`}
              >
                7 Days
              </button>
              <button
                onClick={() => setTimeRange(30)}
                className={`px-4 py-2 rounded-[var(--radius)] font-medium whitespace-nowrap ${
                  timeRange === 30
                    ? "bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] text-white"
                    : "glass-hover text-[var(--color-text)]"
                }`}
              >
                30 Days
              </button>
              <button
                onClick={() => setTimeRange(90)}
                className={`px-4 py-2 rounded-[var(--radius)] font-medium whitespace-nowrap ${
                  timeRange === 90
                    ? "bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] text-white"
                    : "glass-hover text-[var(--color-text)]"
                }`}
              >
                90 Days
              </button>
              <button
                onClick={() => setTimeRange(365)}
                className={`px-4 py-2 rounded-[var(--radius)] font-medium whitespace-nowrap ${
                  timeRange === 365
                    ? "bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] text-white"
                    : "glass-hover text-[var(--color-text)]"
                }`}
              >
                1 Year
              </button>
            </div>
          </div>

          {/* Sleep Trends */}
          {chartData.sleep.length > 0 && (
            <div className="glass rounded-[var(--radius-lg)] p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-primary-dark)] flex items-center justify-center">
                  <Moon className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-[var(--color-text)]">Sleep Trends</h2>
                  <p className="text-sm text-[var(--color-text-secondary)]">Duration and quality over time</p>
                </div>
              </div>

              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData.sleep}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                    <XAxis dataKey="date" stroke="var(--color-text-secondary)" fontSize={12} />
                    <YAxis stroke="var(--color-text-secondary)" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(255, 255, 255, 0.95)",
                        border: "1px solid rgba(0,0,0,0.1)",
                        borderRadius: "8px",
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="hours"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      name="Sleep Hours"
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="quality"
                      stroke="#8b5cf6"
                      strokeWidth={2}
                      name="Quality (1-10)"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Mood & Stress Trends */}
          {chartData.mood.length > 0 && (
            <div className="glass rounded-[var(--radius-lg)] p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-secondary)] to-[var(--color-secondary-dark)] flex items-center justify-center">
                  <Heart className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-[var(--color-text)]">Mood & Stress Trends</h2>
                  <p className="text-sm text-[var(--color-text-secondary)]">Emotional wellbeing patterns</p>
                </div>
              </div>

              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData.mood}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                    <XAxis dataKey="date" stroke="var(--color-text-secondary)" fontSize={12} />
                    <YAxis stroke="var(--color-text-secondary)" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(255, 255, 255, 0.95)",
                        border: "1px solid rgba(0,0,0,0.1)",
                        borderRadius: "8px",
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="mood"
                      stroke="#ec4899"
                      strokeWidth={2}
                      name="Mood Score"
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="stress"
                      stroke="#f97316"
                      strokeWidth={2}
                      name="Stress Level"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Symptoms Trends */}
          {chartData.symptoms.length > 0 && (
            <div className="glass rounded-[var(--radius-lg)] p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center">
                  <Activity className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-[var(--color-text)]">Symptoms Trends</h2>
                  <p className="text-sm text-[var(--color-text-secondary)]">Track symptom severity over time</p>
                </div>
              </div>

              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData.symptoms}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                    <XAxis dataKey="date" stroke="var(--color-text-secondary)" fontSize={12} />
                    <YAxis stroke="var(--color-text-secondary)" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(255, 255, 255, 0.95)",
                        border: "1px solid rgba(0,0,0,0.1)",
                        borderRadius: "8px",
                      }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="gi" stroke="#f97316" strokeWidth={2} name="GI Flare" dot={false} />
                    <Line
                      type="monotone"
                      dataKey="skin"
                      stroke="#ec4899"
                      strokeWidth={2}
                      name="Skin Flare"
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="migraine"
                      stroke="#8b5cf6"
                      strokeWidth={2}
                      name="Migraine"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Weekly Averages Comparison */}
          {weeklyAverages.length > 0 && (
            <div className="glass rounded-[var(--radius-lg)] p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent-dark)] flex items-center justify-center">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-[var(--color-text)]">Weekly Averages</h2>
                  <p className="text-sm text-[var(--color-text-secondary)]">Compare your progress week by week</p>
                </div>
              </div>

              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={weeklyAverages}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                    <XAxis dataKey="week" stroke="var(--color-text-secondary)" fontSize={12} />
                    <YAxis stroke="var(--color-text-secondary)" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(255, 255, 255, 0.95)",
                        border: "1px solid rgba(0,0,0,0.1)",
                        borderRadius: "8px",
                      }}
                    />
                    <Legend />
                    <Bar dataKey="sleep" fill="#3b82f6" name="Avg Sleep (hrs)" radius={[8, 8, 0, 0]} />
                    <Bar dataKey="mood" fill="#ec4899" name="Avg Mood" radius={[8, 8, 0, 0]} />
                    <Bar dataKey="stress" fill="#f97316" name="Avg Stress" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="glass rounded-[var(--radius-lg)] p-6">
              <h3 className="text-sm font-semibold text-[var(--color-text-secondary)] mb-4">
                {timeRange}-Day Averages
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-[var(--color-text)]">Sleep Duration</span>
                  <span className="text-lg font-bold text-[var(--color-text)]">{stats.avgSleep} hrs</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-[var(--color-text)]">Sleep Quality</span>
                  <span className="text-lg font-bold text-[var(--color-text)]">{stats.avgQuality}/10</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-[var(--color-text)]">Mood Score</span>
                  <span className="text-lg font-bold text-[var(--color-text)]">{stats.avgMood}/10</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-[var(--color-text)]">Stress Level</span>
                  <span className="text-lg font-bold text-[var(--color-text)]">{stats.avgStress}/10</span>
                </div>
              </div>
            </div>

            <div className="glass rounded-[var(--radius-lg)] p-6">
              <h3 className="text-sm font-semibold text-[var(--color-text-secondary)] mb-4">Best Days</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-[var(--color-text)]">Best Sleep</span>
                  <span className="text-lg font-bold text-green-600">{stats.bestSleep}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-[var(--color-text)]">Best Mood</span>
                  <span className="text-lg font-bold text-green-600">{stats.bestMood}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-[var(--color-text)]">Lowest Stress</span>
                  <span className="text-lg font-bold text-green-600">{stats.lowestStress}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-[var(--color-text)]">Lowest Symptoms</span>
                  <span className="text-lg font-bold text-green-600">{stats.lowestSymptoms}</span>
                </div>
              </div>
            </div>

            <div className="glass rounded-[var(--radius-lg)] p-6">
              <h3 className="text-sm font-semibold text-[var(--color-text-secondary)] mb-4">Progress</h3>
              <div className="text-center py-8">
                <p className="text-sm text-[var(--color-text-secondary)] mb-2">Keep logging data to see</p>
                <p className="text-sm text-[var(--color-text-secondary)]">your improvement trends</p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
