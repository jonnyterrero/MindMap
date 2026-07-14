"use client";

import { useEffect, useState } from "react";
import HealthMetricsCard from "@/components/HealthMetricsCard";
import { Moon, Heart, Brain, Flame } from "lucide-react";

interface HealthLog {
  date: string;
  sleep_quality_score: number;
  mood_score: number;
  stress_score: number;
}

export default function DashboardMetrics() {
  const [metrics, setMetrics] = useState({
    sleepQuality: 0,
    moodScore: 0,
    stressLevel: 0,
    streak: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await fetch("/api/health-logs");
      if (response.ok) {
        const data = await response.json();
        const logs: HealthLog[] = data.logs || [];

        if (logs.length > 0) {
          // Calculate averages from recent logs
          const recentLogs = logs.slice(0, 7); // Last 7 days
          const avgSleep = recentLogs.reduce((sum, log) => sum + (log.sleep_quality_score || 0), 0) / recentLogs.length;
          const avgMood = recentLogs.reduce((sum, log) => sum + (log.mood_score || 0), 0) / recentLogs.length;
          const avgStress = recentLogs.reduce((sum, log) => sum + (log.stress_score || 0), 0) / recentLogs.length;

          // Calculate streak
          const sortedLogs = [...logs].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
          let streak = 0;
          const today = new Date();
          today.setHours(0, 0, 0, 0);

          for (let i = 0; i < sortedLogs.length; i++) {
            const logDate = new Date(sortedLogs[i].date);
            logDate.setHours(0, 0, 0, 0);
            const daysDiff = Math.floor((today.getTime() - logDate.getTime()) / (1000 * 60 * 60 * 24));

            if (daysDiff === i) {
              streak++;
            } else {
              break;
            }
          }

          setMetrics({
            sleepQuality: Math.round(avgSleep * 10) / 10,
            moodScore: Math.round(avgMood * 10) / 10,
            stressLevel: Math.round(avgStress * 10) / 10,
            streak,
          });
        }
      }
    } catch (error) {
      console.error("Error fetching metrics:", error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-32 bg-muted animate-pulse rounded-lg" />
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <HealthMetricsCard
        icon={Moon}
        title="Sleep Quality"
        value={metrics.sleepQuality > 0 ? metrics.sleepQuality.toFixed(1) : "—"}
        maxValue="10"
        trend={0}
        iconColor="text-blue-600"
        iconBg="bg-blue-100"
      />
      <HealthMetricsCard
        icon={Heart}
        title="Mood Score"
        value={metrics.moodScore > 0 ? metrics.moodScore.toFixed(1) : "—"}
        maxValue="10"
        trend={0}
        iconColor="text-pink-600"
        iconBg="bg-pink-100"
      />
      <HealthMetricsCard
        icon={Brain}
        title="Stress Level"
        value={metrics.stressLevel > 0 ? metrics.stressLevel.toFixed(1) : "—"}
        maxValue="10"
        trend={0}
        iconColor="text-purple-600"
        iconBg="bg-purple-100"
      />
      <HealthMetricsCard
        icon={Flame}
        title="Current Streak"
        value={metrics.streak.toString()}
        maxValue="days"
        trend={0}
        iconColor="text-orange-600"
        iconBg="bg-orange-100"
      />
    </div>
  );
}