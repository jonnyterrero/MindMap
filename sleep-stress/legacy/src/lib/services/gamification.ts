// Gamification Service - Badges, Achievements, Streaks, and XP

import type { HealthLog } from './insights';

export interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  color: string;
  category: 'streak' | 'milestone' | 'improvement';
  progress: number;
  maxProgress: number;
  isUnlocked: boolean;
  unlockedAt?: string;
}

export interface Achievement {
  id: string;
  title: string;
  description: string;
  points: number;
  isUnlocked: boolean;
  unlockedAt?: string;
}

export interface UserStats {
  totalLogs: number;
  totalDays: number;
  averageMood: number;
  averageSleep: number;
  averageStress: number;
  badges: Badge[];
  achievements: Achievement[];
  streaks: {
    currentStreak: number;
    longestStreak: number;
    lastLogDate: string;
  };
  level: number;
  experience: number;
  nextLevelExp: number;
}

export class GamificationService {
  private badges: Badge[] = [
    {
      id: 'streak_3',
      name: 'Getting Started',
      description: 'Log 3 days in a row',
      icon: '🔥',
      color: '#FF6B6B',
      category: 'streak',
      progress: 0,
      maxProgress: 3,
      isUnlocked: false
    },
    {
      id: 'streak_7',
      name: 'Week Warrior',
      description: 'Log 7 days in a row',
      icon: '🏆',
      color: '#4ECDC4',
      category: 'streak',
      progress: 0,
      maxProgress: 7,
      isUnlocked: false
    },
    {
      id: 'streak_30',
      name: 'Monthly Master',
      description: 'Log 30 days in a row',
      icon: '🏅',
      color: '#45B7D1',
      category: 'streak',
      progress: 0,
      maxProgress: 30,
      isUnlocked: false
    },
    {
      id: 'milestone_50',
      name: 'Half Century',
      description: 'Complete 50 health logs',
      icon: '✓',
      color: '#FF6348',
      category: 'milestone',
      progress: 0,
      maxProgress: 50,
      isUnlocked: false
    },
    {
      id: 'mood_improvement',
      name: 'Mood Booster',
      description: 'Improve mood score by 2+ points over 7 days',
      icon: '😊',
      color: '#54A0FF',
      category: 'improvement',
      progress: 0,
      maxProgress: 1,
      isUnlocked: false
    }
  ];

  private achievements: Achievement[] = [
    {
      id: 'first_log',
      title: 'First Steps',
      description: 'Complete your first health log',
      points: 10,
      isUnlocked: false
    },
    {
      id: 'week_complete',
      title: 'Week Complete',
      description: 'Log every day for a full week',
      points: 50,
      isUnlocked: false
    },
    {
      id: 'mood_master',
      title: 'Mood Master',
      description: 'Maintain average mood above 8 for a week',
      points: 100,
      isUnlocked: false
    }
  ];

  updateStats(healthData: HealthLog[]): UserStats {
    const stats = this.calculateStats(healthData);
    this.checkBadges(healthData, stats);
    this.checkAchievements(healthData, stats);
    return stats;
  }

  private calculateStats(healthData: HealthLog[]): UserStats {
    const totalLogs = healthData.length;
    const dates = healthData.map(d => d.date);
    const uniqueDates = new Set(dates);
    const totalDays = uniqueDates.size;

    // Calculate averages
    const moodScores = healthData.filter(d => d.moodScore != null).map(d => d.moodScore!);
    const sleepScores = healthData.filter(d => d.sleepQualityScore != null).map(d => d.sleepQualityScore!);
    const stressScores = healthData.filter(d => d.stressScore != null).map(d => d.stressScore!);

    const averageMood = moodScores.length > 0 ? moodScores.reduce((a, b) => a + b, 0) / moodScores.length : 0;
    const averageSleep = sleepScores.length > 0 ? sleepScores.reduce((a, b) => a + b, 0) / sleepScores.length : 0;
    const averageStress = stressScores.length > 0 ? stressScores.reduce((a, b) => a + b, 0) / stressScores.length : 0;

    // Calculate streaks
    const streaks = this.calculateStreaks(healthData);

    // Calculate level and experience
    const experience = totalLogs * 10; // 10 XP per log
    const level = Math.floor(experience / 100) + 1; // 100 XP per level
    const nextLevelExp = level * 100;

    return {
      totalLogs,
      totalDays,
      averageMood,
      averageSleep,
      averageStress,
      badges: [...this.badges],
      achievements: [...this.achievements],
      streaks,
      level,
      experience,
      nextLevelExp
    };
  }

  private calculateStreaks(healthData: HealthLog[]): { currentStreak: number; longestStreak: number; lastLogDate: string } {
    if (healthData.length === 0) {
      return { currentStreak: 0, longestStreak: 0, lastLogDate: '' };
    }

    const sortedData = [...healthData].sort((a, b) => 
      new Date(a.date).getTime() - new Date(b.date).getTime()
    );
    const dates = Array.from(new Set(sortedData.map(d => d.date))).sort();

    // Calculate current streak
    let currentStreak = 0;
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    for (let i = dates.length - 1; i >= 0; i--) {
      const logDate = new Date(dates[i]);
      logDate.setHours(0, 0, 0, 0);
      
      const daysDiff = Math.floor((today.getTime() - logDate.getTime()) / (1000 * 60 * 60 * 24));
      
      if (daysDiff === currentStreak) {
        currentStreak++;
      } else {
        break;
      }
    }

    // Calculate longest streak
    let longestStreak = 0;
    let tempStreak = 1;

    for (let i = 1; i < dates.length; i++) {
      const prevDate = new Date(dates[i - 1]);
      const currDate = new Date(dates[i]);
      const daysDiff = Math.floor((currDate.getTime() - prevDate.getTime()) / (1000 * 60 * 60 * 24));

      if (daysDiff === 1) {
        tempStreak++;
      } else {
        longestStreak = Math.max(longestStreak, tempStreak);
        tempStreak = 1;
      }
    }
    longestStreak = Math.max(longestStreak, tempStreak);

    return {
      currentStreak,
      longestStreak,
      lastLogDate: dates[dates.length - 1] || ''
    };
  }

  private checkBadges(healthData: HealthLog[], stats: UserStats): void {
    this.badges.forEach(badge => {
      if (badge.isUnlocked) return;

      let shouldUnlock = false;

      if (badge.id.startsWith('streak_')) {
        const requiredStreak = parseInt(badge.id.split('_')[1]);
        badge.progress = Math.min(stats.streaks.currentStreak, badge.maxProgress);
        shouldUnlock = stats.streaks.currentStreak >= requiredStreak;
      } else if (badge.id.startsWith('milestone_')) {
        const requiredLogs = parseInt(badge.id.split('_')[1]);
        badge.progress = Math.min(stats.totalLogs, badge.maxProgress);
        shouldUnlock = stats.totalLogs >= requiredLogs;
      } else if (badge.id === 'mood_improvement') {
        const recentWeek = healthData.slice(-7).filter(d => d.moodScore != null);
        if (recentWeek.length >= 7) {
          const firstMood = recentWeek[0].moodScore!;
          const lastMood = recentWeek[recentWeek.length - 1].moodScore!;
          shouldUnlock = lastMood - firstMood >= 2;
          badge.progress = shouldUnlock ? 1 : 0;
        }
      }

      if (shouldUnlock) {
        badge.isUnlocked = true;
        badge.unlockedAt = new Date().toISOString();
      }
    });
  }

  private checkAchievements(healthData: HealthLog[], stats: UserStats): void {
    this.achievements.forEach(achievement => {
      if (achievement.isUnlocked) return;

      let shouldUnlock = false;

      switch (achievement.id) {
        case 'first_log':
          shouldUnlock = stats.totalLogs >= 1;
          break;
        case 'week_complete':
          shouldUnlock = stats.streaks.currentStreak >= 7;
          break;
        case 'mood_master':
          shouldUnlock = stats.averageMood >= 8;
          break;
      }

      if (shouldUnlock) {
        achievement.isUnlocked = true;
        achievement.unlockedAt = new Date().toISOString();
      }
    });
  }

  getBadges(): Badge[] {
    return [...this.badges];
  }

  getAchievements(): Achievement[] {
    return [...this.achievements];
  }
}