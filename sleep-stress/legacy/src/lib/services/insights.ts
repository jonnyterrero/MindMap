// Personalized Insights Service - Analyzes health data for patterns and recommendations

export interface HealthLog {
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

export interface Insight {
  id: string;
  type: 'sleep' | 'mood' | 'stress' | 'correlation' | 'lifestyle';
  title: string;
  description: string;
  actionable: boolean;
  priority: 'low' | 'medium' | 'high';
  confidence: number;
  recommendations?: string[];
  category: 'warning' | 'achievement' | 'tip' | 'correlation';
}

export class InsightsService {
  private insights: Insight[] = [];

  generateInsights(healthData: HealthLog[]): Insight[] {
    this.insights = [];

    if (healthData.length < 7) {
      return this.insights;
    }

    // Sort data by date
    const sortedData = [...healthData].sort((a, b) => 
      new Date(a.date).getTime() - new Date(b.date).getTime()
    );

    // Analyze different aspects
    this.analyzeSleepPatterns(sortedData);
    this.analyzeMoodTrends(sortedData);
    this.analyzeStressPatterns(sortedData);
    this.analyzeSymptomCorrelations(sortedData);
    this.analyzeLifestyleFactors(sortedData);

    // Sort by priority and confidence
    return this.insights.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];
      if (priorityDiff !== 0) return priorityDiff;
      return b.confidence - a.confidence;
    });
  }

  private analyzeSleepPatterns(data: HealthLog[]): void {
    const sleepData = data.filter(d => d.sleepDurationHours != null && d.sleepQualityScore != null);
    if (sleepData.length < 7) return;

    const durations = sleepData.map(d => d.sleepDurationHours!);
    const qualities = sleepData.map(d => d.sleepQualityScore!);
    
    const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
    const recent7 = durations.slice(-7);
    const recentAvg = recent7.reduce((a, b) => a + b, 0) / recent7.length;
    const avgQuality = qualities.reduce((a, b) => a + b, 0) / qualities.length;

    // Low sleep duration alert
    if (avgDuration < 6.5) {
      this.insights.push({
        id: 'sleep_duration_low',
        type: 'sleep',
        title: 'Sleep Duration Alert',
        description: `Your average sleep duration is ${avgDuration.toFixed(1)} hours, which is below the recommended 7-9 hours.`,
        actionable: true,
        priority: 'high',
        confidence: 0.9,
        recommendations: [
          'Try going to bed 30 minutes earlier',
          'Create a consistent bedtime routine',
          'Avoid screens 1 hour before bed',
          'Keep your bedroom cool and dark'
        ],
        category: 'warning'
      });
    }

    // Sleep quality improving
    const recentQuality = qualities.slice(-7).reduce((a, b) => a + b, 0) / Math.min(7, qualities.length);
    if (recentQuality > avgQuality + 1) {
      this.insights.push({
        id: 'sleep_quality_improving',
        type: 'sleep',
        title: 'Sleep Quality Improving!',
        description: `Your sleep quality has improved from ${avgQuality.toFixed(1)} to ${recentQuality.toFixed(1)} over the past week.`,
        actionable: false,
        priority: 'low',
        confidence: 0.8,
        category: 'achievement'
      });
    }

    // Sleep consistency
    const variance = durations.reduce((sum, val) => sum + Math.pow(val - avgDuration, 2), 0) / durations.length;
    const stdDev = Math.sqrt(variance);
    if (stdDev > 1.5) {
      this.insights.push({
        id: 'sleep_inconsistent',
        type: 'sleep',
        title: 'Inconsistent Sleep Schedule',
        description: `Your sleep duration varies significantly (±${stdDev.toFixed(1)}h). Consistency is key for quality rest.`,
        actionable: true,
        priority: 'medium',
        confidence: 0.85,
        recommendations: [
          'Set a consistent bedtime and wake time',
          'Avoid long naps during the day',
          'Create a relaxing pre-sleep routine'
        ],
        category: 'tip'
      });
    }
  }

  private analyzeMoodTrends(data: HealthLog[]): void {
    const moodData = data.filter(d => d.moodScore != null);
    if (moodData.length < 7) return;

    const moods = moodData.map(d => d.moodScore!);
    const avgMood = moods.reduce((a, b) => a + b, 0) / moods.length;
    const recent7 = moods.slice(-7);
    const recentMood = recent7.reduce((a, b) => a + b, 0) / recent7.length;

    // Declining mood trend
    if (recentMood < avgMood - 1.5) {
      this.insights.push({
        id: 'mood_declining',
        type: 'mood',
        title: 'Mood Trend Alert',
        description: `Your mood has declined from ${avgMood.toFixed(1)} to ${recentMood.toFixed(1)} over the past week.`,
        actionable: true,
        priority: 'high',
        confidence: 0.8,
        recommendations: [
          'Consider talking to a mental health professional',
          'Try daily gratitude journaling',
          'Increase physical activity',
          'Connect with friends and family'
        ],
        category: 'warning'
      });
    }

    // Consistently good mood
    if (avgMood >= 7.5) {
      this.insights.push({
        id: 'mood_positive',
        type: 'mood',
        title: 'Excellent Mood Management',
        description: `Your average mood score is ${avgMood.toFixed(1)}/10. You're maintaining positive mental health!`,
        actionable: false,
        priority: 'low',
        confidence: 0.9,
        category: 'achievement'
      });
    }
  }

  private analyzeStressPatterns(data: HealthLog[]): void {
    const stressData = data.filter(d => d.stressScore != null);
    if (stressData.length < 7) return;

    const stressLevels = stressData.map(d => d.stressScore!);
    const avgStress = stressLevels.reduce((a, b) => a + b, 0) / stressLevels.length;

    // High stress alert
    if (avgStress > 7) {
      this.insights.push({
        id: 'high_stress',
        type: 'stress',
        title: 'High Stress Levels',
        description: `Your average stress level is ${avgStress.toFixed(1)}/10, which is quite high.`,
        actionable: true,
        priority: 'high',
        confidence: 0.9,
        recommendations: [
          'Practice deep breathing exercises',
          'Try progressive muscle relaxation',
          'Consider stress management therapy',
          'Identify and address stress sources'
        ],
        category: 'warning'
      });
    }

    // Stress increasing trend
    const recent7 = stressLevels.slice(-7);
    const recentStress = recent7.reduce((a, b) => a + b, 0) / recent7.length;
    if (recentStress > avgStress + 1) {
      this.insights.push({
        id: 'stress_increasing',
        type: 'stress',
        title: 'Rising Stress Levels',
        description: `Your stress has increased from ${avgStress.toFixed(1)} to ${recentStress.toFixed(1)} recently.`,
        actionable: true,
        priority: 'medium',
        confidence: 0.75,
        recommendations: [
          'Take regular breaks during work',
          'Practice mindfulness meditation',
          'Ensure adequate sleep'
        ],
        category: 'warning'
      });
    }
  }

  private analyzeSymptomCorrelations(data: HealthLog[]): void {
    const relevantData = data.filter(d => 
      d.sleepQualityScore != null && d.giFlare != null
    );
    if (relevantData.length < 14) return;

    const sleepScores = relevantData.map(d => d.sleepQualityScore!);
    const giFlares = relevantData.map(d => d.giFlare!);

    // Calculate correlation
    const correlation = this.calculateCorrelation(sleepScores, giFlares);

    if (correlation < -0.4) {
      this.insights.push({
        id: 'sleep_gi_correlation',
        type: 'correlation',
        title: 'Sleep Quality & GI Symptoms',
        description: 'Poor sleep quality is correlated with increased GI flare-ups in your data.',
        actionable: true,
        priority: 'medium',
        confidence: Math.abs(correlation),
        recommendations: [
          'Prioritize sleep quality to reduce GI symptoms',
          'Try sleep hygiene improvements',
          'Consider sleep environment optimization'
        ],
        category: 'correlation'
      });
    }
  }

  private analyzeLifestyleFactors(data: HealthLog[]): void {
    const datedData = data.map(d => ({
      ...d,
      dayOfWeek: new Date(d.date).getDay()
    }));

    const weekendData = datedData.filter(d => d.dayOfWeek === 0 || d.dayOfWeek === 6);
    const weekdayData = datedData.filter(d => d.dayOfWeek >= 1 && d.dayOfWeek <= 5);

    if (weekendData.length >= 4 && weekdayData.length >= 10) {
      const weekendSleep = weekendData
        .filter(d => d.sleepDurationHours != null)
        .map(d => d.sleepDurationHours!);
      const weekdaySleep = weekdayData
        .filter(d => d.sleepDurationHours != null)
        .map(d => d.sleepDurationHours!);

      if (weekendSleep.length > 0 && weekdaySleep.length > 0) {
        const weekendAvg = weekendSleep.reduce((a, b) => a + b, 0) / weekendSleep.length;
        const weekdayAvg = weekdaySleep.reduce((a, b) => a + b, 0) / weekdaySleep.length;

        if (weekendAvg > weekdayAvg + 1) {
          this.insights.push({
            id: 'weekend_sleep_catchup',
            type: 'lifestyle',
            title: 'Weekend Sleep Catch-up',
            description: `You sleep ${(weekendAvg - weekdayAvg).toFixed(1)} hours more on weekends, suggesting weekday sleep debt.`,
            actionable: true,
            priority: 'medium',
            confidence: 0.8,
            recommendations: [
              'Try to get more consistent sleep during weekdays',
              'Consider adjusting weekday bedtime',
              'Avoid oversleeping on weekends'
            ],
            category: 'tip'
          });
        }
      }
    }
  }

  private calculateCorrelation(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length);
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    return denominator === 0 ? 0 : numerator / denominator;
  }
}