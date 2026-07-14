"use client"

import { Trophy, Flame, Star, Award } from "lucide-react"
import { useHealthData } from "@/contexts/HealthDataContext"

export function GamificationWidget() {
  const { gamificationData } = useHealthData()

  const xpPercentage = (gamificationData.xp / gamificationData.xpToNextLevel) * 100
  const earnedBadges = gamificationData.badges.filter((b) => b.earned).length
  const latestAchievement = gamificationData.achievements[gamificationData.achievements.length - 1]

  return (
    <div className="glass rounded-[var(--radius-lg)] p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-[var(--color-text)]">Your Progress</h3>
        <Trophy className="w-5 h-5 text-[var(--color-primary)]" />
      </div>

      {/* Level Progress */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-[var(--color-text-secondary)]">Level {gamificationData.level}</span>
          <span className="text-sm font-medium text-[var(--color-text)]">
            {gamificationData.xp}/{gamificationData.xpToNextLevel} XP
          </span>
        </div>
        <div className="h-3 bg-white/50 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] rounded-full transition-all duration-500"
            style={{ width: `${xpPercentage}%` }}
          />
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-gradient-to-br from-[var(--color-glass-purple)] to-white/50 rounded-[var(--radius)] p-3">
          <div className="flex items-center gap-2 mb-1">
            <Flame className="w-4 h-4 text-orange-500" />
            <span className="text-xs text-[var(--color-text-secondary)]">Streak</span>
          </div>
          <p className="text-2xl font-bold text-[var(--color-text)]">{gamificationData.currentStreak}</p>
          <p className="text-xs text-[var(--color-text-muted)]">days</p>
        </div>

        <div className="bg-gradient-to-br from-[var(--color-glass-pink)] to-white/50 rounded-[var(--radius)] p-3">
          <div className="flex items-center gap-2 mb-1">
            <Star className="w-4 h-4 text-yellow-500" />
            <span className="text-xs text-[var(--color-text-secondary)]">Badges</span>
          </div>
          <p className="text-2xl font-bold text-[var(--color-text)]">{earnedBadges}</p>
          <p className="text-xs text-[var(--color-text-muted)]\">earned</p>
        </div>
      </div>

      {/* Recent Badge */}
      {latestAchievement ? (
        <div className="bg-gradient-to-r from-[var(--color-glass-blue)] to-[var(--color-glass-purple)] rounded-[var(--radius)] p-4">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full bg-white/80 flex items-center justify-center">
              <Award className="w-6 h-6 text-[var(--color-primary)]" />
            </div>
            <div>
              <p className="text-sm font-semibold text-[var(--color-text)]">{latestAchievement.name}</p>
              <p className="text-xs text-[var(--color-text-secondary)]">{latestAchievement.description}</p>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-gradient-to-r from-[var(--color-glass-blue)] to-[var(--color-glass-purple)] rounded-[var(--radius)] p-4">
          <div className="text-center">
            <Award className="w-8 h-8 text-[var(--color-primary)] mx-auto mb-2" />
            <p className="text-sm font-semibold text-[var(--color-text)]">Start Your Journey</p>
            <p className="text-xs text-[var(--color-text-secondary)]">Log data to earn achievements!</p>
          </div>
        </div>
      )}
    </div>
  )
}
