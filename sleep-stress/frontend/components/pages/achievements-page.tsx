"use client"

import { Trophy, Flame, Star, Award, Medal, Target, Zap, Crown } from "lucide-react"

export function AchievementsPage() {
  const userStats = {
    level: 5,
    experience: 450,
    nextLevelExp: 500,
    totalLogs: 45,
    currentStreak: 12,
    longestStreak: 18,
    totalBadges: 8,
    totalAchievements: 12,
  }

  const badges = [
    {
      id: "streak_3",
      name: "Getting Started",
      description: "Log 3 days in a row",
      icon: <Flame className="w-6 h-6" />,
      color: "from-orange-400 to-red-500",
      isUnlocked: true,
      progress: 3,
      maxProgress: 3,
      unlockedAt: "2024-01-15",
    },
    {
      id: "streak_7",
      name: "Week Warrior",
      description: "Log 7 days in a row",
      icon: <Trophy className="w-6 h-6" />,
      color: "from-blue-400 to-cyan-500",
      isUnlocked: true,
      progress: 7,
      maxProgress: 7,
      unlockedAt: "2024-01-20",
    },
    {
      id: "streak_30",
      name: "Monthly Master",
      description: "Log 30 days in a row",
      icon: <Medal className="w-6 h-6" />,
      color: "from-purple-400 to-purple-600",
      isUnlocked: false,
      progress: 12,
      maxProgress: 30,
    },
    {
      id: "milestone_50",
      name: "Half Century",
      description: "Complete 50 health logs",
      icon: <Star className="w-6 h-6" />,
      color: "from-yellow-400 to-orange-500",
      isUnlocked: false,
      progress: 45,
      maxProgress: 50,
    },
    {
      id: "mood_improvement",
      name: "Mood Booster",
      description: "Improve mood score by 2+ points over 7 days",
      icon: <Zap className="w-6 h-6" />,
      color: "from-pink-400 to-pink-600",
      isUnlocked: true,
      progress: 1,
      maxProgress: 1,
      unlockedAt: "2024-01-22",
    },
    {
      id: "sleep_master",
      name: "Sleep Master",
      description: "Maintain 8+ hours sleep for 7 days",
      icon: <Crown className="w-6 h-6" />,
      color: "from-indigo-400 to-indigo-600",
      isUnlocked: false,
      progress: 4,
      maxProgress: 7,
    },
  ]

  const achievements = [
    {
      id: "first_log",
      title: "First Steps",
      description: "Complete your first health log",
      points: 10,
      isUnlocked: true,
      unlockedAt: "2024-01-10",
    },
    {
      id: "week_complete",
      title: "Week Complete",
      description: "Log every day for a full week",
      points: 50,
      isUnlocked: true,
      unlockedAt: "2024-01-20",
    },
    {
      id: "mood_master",
      title: "Mood Master",
      description: "Maintain average mood above 8 for a week",
      points: 100,
      isUnlocked: false,
    },
    {
      id: "consistency_king",
      title: "Consistency King",
      description: "Log for 30 consecutive days",
      points: 200,
      isUnlocked: false,
    },
  ]

  const progressPercentage = (userStats.experience / userStats.nextLevelExp) * 100

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="glass rounded-[var(--radius-lg)] p-6 md:p-8">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-14 h-14 rounded-[var(--radius)] bg-gradient-to-br from-yellow-400 to-orange-500 flex items-center justify-center">
            <Trophy className="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 className="text-3xl md:text-4xl font-bold gradient-text">Achievements</h1>
            <p className="text-[var(--color-text-secondary)]">Track your progress and unlock rewards</p>
          </div>
        </div>

        {/* Level Progress */}
        <div className="bg-gradient-to-r from-[var(--color-glass-purple)] to-[var(--color-glass-blue)] rounded-[var(--radius)] p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-[var(--color-text-secondary)] mb-1">Current Level</p>
              <p className="text-4xl font-bold text-[var(--color-text)]">{userStats.level}</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-[var(--color-text-secondary)] mb-1">Experience</p>
              <p className="text-2xl font-bold text-[var(--color-text)]">
                {userStats.experience}/{userStats.nextLevelExp} XP
              </p>
            </div>
          </div>
          <div className="h-4 bg-white/50 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] rounded-full transition-all duration-500"
              style={{ width: `${progressPercentage}%` }}
            />
          </div>
          <p className="text-xs text-[var(--color-text-secondary)] mt-2 text-center">
            {userStats.nextLevelExp - userStats.experience} XP until Level {userStats.level + 1}
          </p>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass rounded-[var(--radius-lg)] p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-[var(--radius)] bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center">
              <Flame className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-xs text-[var(--color-text-secondary)]">Current Streak</p>
              <p className="text-2xl font-bold text-[var(--color-text)]">{userStats.currentStreak}</p>
            </div>
          </div>
          <p className="text-xs text-[var(--color-text-muted)]">days in a row</p>
        </div>

        <div className="glass rounded-[var(--radius-lg)] p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-[var(--radius)] bg-gradient-to-br from-purple-400 to-purple-600 flex items-center justify-center">
              <Target className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-xs text-[var(--color-text-secondary)]">Longest Streak</p>
              <p className="text-2xl font-bold text-[var(--color-text)]">{userStats.longestStreak}</p>
            </div>
          </div>
          <p className="text-xs text-[var(--color-text-muted)]">days record</p>
        </div>

        <div className="glass rounded-[var(--radius-lg)] p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-[var(--radius)] bg-gradient-to-br from-blue-400 to-cyan-500 flex items-center justify-center">
              <Star className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-xs text-[var(--color-text-secondary)]">Badges</p>
              <p className="text-2xl font-bold text-[var(--color-text)]">{userStats.totalBadges}</p>
            </div>
          </div>
          <p className="text-xs text-[var(--color-text-muted)]">earned</p>
        </div>

        <div className="glass rounded-[var(--radius-lg)] p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-[var(--radius)] bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center">
              <Award className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-xs text-[var(--color-text-secondary)]">Total Logs</p>
              <p className="text-2xl font-bold text-[var(--color-text)]">{userStats.totalLogs}</p>
            </div>
          </div>
          <p className="text-xs text-[var(--color-text-muted)]">entries</p>
        </div>
      </div>

      {/* Badges Section */}
      <div className="glass rounded-[var(--radius-lg)] p-6">
        <h2 className="text-2xl font-bold text-[var(--color-text)] mb-6">Badges</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {badges.map((badge) => (
            <div
              key={badge.id}
              className={`glass-hover rounded-[var(--radius)] p-5 ${badge.isUnlocked ? "" : "opacity-60"}`}
            >
              <div className="flex items-start gap-4 mb-4">
                <div
                  className={`w-14 h-14 rounded-[var(--radius)] bg-gradient-to-br ${badge.color} flex items-center justify-center text-white ${badge.isUnlocked ? "" : "grayscale"}`}
                >
                  {badge.icon}
                </div>
                <div className="flex-1">
                  <h3 className="font-bold text-[var(--color-text)] mb-1">{badge.name}</h3>
                  <p className="text-xs text-[var(--color-text-secondary)]">{badge.description}</p>
                </div>
              </div>

              {badge.isUnlocked ? (
                <div className="flex items-center gap-2 text-xs text-green-600">
                  <div className="w-2 h-2 rounded-full bg-green-500" />
                  <span>Unlocked on {badge.unlockedAt}</span>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[var(--color-text-secondary)]">Progress</span>
                    <span className="font-medium text-[var(--color-text)]">
                      {badge.progress}/{badge.maxProgress}
                    </span>
                  </div>
                  <div className="h-2 bg-white/50 rounded-full overflow-hidden">
                    <div
                      className={`h-full bg-gradient-to-r ${badge.color} rounded-full`}
                      style={{ width: `${(badge.progress / badge.maxProgress) * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Achievements Section */}
      <div className="glass rounded-[var(--radius-lg)] p-6">
        <h2 className="text-2xl font-bold text-[var(--color-text)] mb-6">Achievements</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {achievements.map((achievement) => (
            <div
              key={achievement.id}
              className={`glass-hover rounded-[var(--radius)] p-5 flex items-start gap-4 ${achievement.isUnlocked ? "border-l-4 border-green-500" : "opacity-60"}`}
            >
              <div
                className={`w-12 h-12 rounded-[var(--radius)] ${achievement.isUnlocked ? "bg-gradient-to-br from-yellow-400 to-orange-500" : "bg-gray-300"} flex items-center justify-center text-white`}
              >
                <Award className="w-6 h-6" />
              </div>
              <div className="flex-1">
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-bold text-[var(--color-text)]">{achievement.title}</h3>
                  <span className="text-sm font-bold text-[var(--color-primary)]">{achievement.points} XP</span>
                </div>
                <p className="text-sm text-[var(--color-text-secondary)] mb-2">{achievement.description}</p>
                {achievement.isUnlocked && (
                  <p className="text-xs text-green-600">Unlocked on {achievement.unlockedAt}</p>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
