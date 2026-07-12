"use client";

import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Brain, 
  TrendingUp, 
  Activity, 
  AlertTriangle, 
  Trophy, 
  Target,
  Sparkles,
  Flame
} from "lucide-react";
import { useEffect, useState } from "react";
import { InsightsService, type Insight } from "@/lib/services/insights";
import { GamificationService, type UserStats } from "@/lib/services/gamification";

interface HealthLog {
  id: number;
  userId: string;
  date: string;
  sleepDurationHours?: number | null;
  sleepQualityScore?: number | null;
  moodScore?: number | null;
  stressScore?: number | null;
  giFlare?: number | null;
  skinFlare?: number | null;
  migraine?: number | null;
  journalEntry?: string | null;
}

export default function AdvancedInsightsPanel() {
  const [loading, setLoading] = useState(true);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [userStats, setUserStats] = useState<UserStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDataAndAnalyze();
  }, []);

  const fetchDataAndAnalyze = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch health logs from API
      const response = await fetch('/api/health-logs?userId=user123&limit=100');
      if (!response.ok) {
        throw new Error('Failed to fetch health logs');
      }

      const healthLogs: HealthLog[] = await response.json();

      // Generate insights
      const insightsService = new InsightsService();
      const generatedInsights = insightsService.generateInsights(healthLogs);
      setInsights(generatedInsights.slice(0, 5)); // Show top 5

      // Calculate gamification stats
      const gamificationService = new GamificationService();
      const stats = gamificationService.updateStats(healthLogs);
      setUserStats(stats);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error analyzing health data:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-5 h-5 text-purple-600 animate-pulse" />
          <h3 className="text-lg font-semibold">AI Health Analysis</h3>
        </div>
        <p className="text-sm text-muted-foreground animate-pulse">
          Analyzing your health data...
        </p>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="w-5 h-5 text-red-600" />
          <h3 className="text-lg font-semibold">Error</h3>
        </div>
        <p className="text-sm text-red-600 mb-4">{error}</p>
        <Button onClick={fetchDataAndAnalyze} size="sm">
          Retry
        </Button>
      </Card>
    );
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'text-red-600 bg-red-50';
      case 'medium': return 'text-orange-600 bg-orange-50';
      default: return 'text-blue-600 bg-blue-50';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'warning': return <AlertTriangle className="w-4 h-4" />;
      case 'achievement': return <Trophy className="w-4 h-4" />;
      case 'tip': return <Sparkles className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Gamification Stats */}
      {userStats && (
        <Card className="p-6 bg-gradient-to-br from-purple-50 to-blue-50">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Trophy className="w-5 h-5 text-purple-600" />
              <h3 className="text-lg font-semibold">Your Progress</h3>
            </div>
            <Badge variant="secondary" className="text-lg">
              Level {userStats.level}
            </Badge>
          </div>

          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="text-center">
              <div className="flex items-center justify-center gap-1 mb-1">
                <Flame className="w-4 h-4 text-orange-500" />
                <p className="text-2xl font-bold">{userStats.streaks.currentStreak}</p>
              </div>
              <p className="text-xs text-muted-foreground">Day Streak</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold">{userStats.totalLogs}</p>
              <p className="text-xs text-muted-foreground">Total Logs</p>
            </div>
          </div>

          {/* XP Progress */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">XP Progress</span>
              <span className="font-medium">{userStats.experience}/{userStats.nextLevelExp}</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="h-2 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 transition-all"
                style={{ width: `${(userStats.experience / userStats.nextLevelExp) * 100}%` }}
              />
            </div>
          </div>

          {/* Unlocked Badges */}
          <div className="mt-4">
            <p className="text-sm font-medium mb-2">Recent Badges:</p>
            <div className="flex flex-wrap gap-2">
              {userStats.badges.filter(b => b.isUnlocked).slice(0, 4).map(badge => (
                <div
                  key={badge.id}
                  className="flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium"
                  style={{ backgroundColor: badge.color + '20', color: badge.color }}
                >
                  <span>{badge.icon}</span>
                  <span>{badge.name}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Average Stats */}
          <div className="mt-4 pt-4 border-t border-purple-200/50">
            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <p className="text-xs text-muted-foreground">Mood</p>
                <p className="text-sm font-semibold">{userStats.averageMood.toFixed(1)}/10</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Sleep</p>
                <p className="text-sm font-semibold">{userStats.averageSleep.toFixed(1)}/10</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Stress</p>
                <p className="text-sm font-semibold">{userStats.averageStress.toFixed(1)}/10</p>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Personalized Insights */}
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-5 h-5 text-purple-600" />
          <h3 className="text-lg font-semibold">Personalized Insights</h3>
        </div>

        {insights.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Keep logging your health data to get personalized insights! Need at least 7 days of data.
          </p>
        ) : (
          <div className="space-y-4">
            {insights.map((insight) => (
              <div
                key={insight.id}
                className="p-4 rounded-lg border bg-card hover:shadow-md transition-shadow"
              >
                <div className="flex items-start gap-3">
                  <div className={`p-2 rounded-full ${getPriorityColor(insight.priority)}`}>
                    {getCategoryIcon(insight.category)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2 mb-1">
                      <h4 className="font-semibold text-sm">{insight.title}</h4>
                      <Badge variant="outline" className="text-xs shrink-0">
                        {Math.round(insight.confidence * 100)}% confident
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      {insight.description}
                    </p>
                    {insight.recommendations && insight.recommendations.length > 0 && (
                      <div className="mt-3 space-y-1">
                        <p className="text-xs font-medium text-muted-foreground">Recommendations:</p>
                        <ul className="space-y-1">
                          {insight.recommendations.slice(0, 2).map((rec, idx) => (
                            <li key={idx} className="text-xs text-muted-foreground flex items-start gap-2">
                              <span className="text-primary mt-0.5">•</span>
                              <span>{rec}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Quick Actions */}
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <Target className="w-5 h-5 text-green-600" />
          <h3 className="text-lg font-semibold">Health Goals</h3>
        </div>
        <p className="text-sm text-muted-foreground mb-4">
          Set and track personalized health goals to improve your wellbeing.
        </p>
        <Button variant="outline" size="sm" className="w-full">
          Set New Goal
        </Button>
      </Card>
    </div>
  );
}