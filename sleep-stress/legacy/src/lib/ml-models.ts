// Mock ML/AI Models for Health Predictions

export interface HealthData {
  sleepQuality: number;
  sleepDuration: number;
  stressLevel: number;
  moodScore: number;
  timestamp: Date;
}

export interface PredictionResult {
  prediction: number;
  confidence: number;
  features: string[];
  model: string;
}

/**
 * Mock Neural Network for Sleep Quality Prediction
 * Simulates a deep learning model trained on sleep patterns
 */
export class SleepQualityPredictor {
  private weights: number[][] = [
    [0.8, -0.6, 0.4],
    [0.5, 0.3, -0.2],
    [-0.3, 0.7, 0.6],
  ];

  predict(data: Partial<HealthData>): PredictionResult {
    const { stressLevel = 5, moodScore = 5, sleepDuration = 7 } = data;
    
    // Simulate neural network forward pass
    const input = [stressLevel / 10, moodScore / 10, sleepDuration / 10];
    let prediction = 0;
    
    for (let i = 0; i < this.weights.length; i++) {
      for (let j = 0; j < input.length; j++) {
        prediction += this.weights[i][j] * input[j];
      }
    }
    
    // Apply activation function (sigmoid)
    prediction = 1 / (1 + Math.exp(-prediction));
    prediction = Math.max(0, Math.min(10, prediction * 10));
    
    return {
      prediction: parseFloat(prediction.toFixed(2)),
      confidence: 0.85 + Math.random() * 0.1,
      features: ['stress_level', 'mood_score', 'sleep_duration'],
      model: 'Deep Neural Network (3-layer)',
    };
  }
}

/**
 * Random Forest Classifier for Symptom Prediction
 * Simulates ensemble learning for symptom occurrence
 */
export class SymptomPredictor {
  private trees = 100;

  predictSymptomRisk(data: Partial<HealthData>): {
    symptom: string;
    risk: number;
    factors: string[];
  }[] {
    const { stressLevel = 5, sleepQuality = 5, moodScore = 5 } = data;
    
    const symptoms = [
      {
        symptom: 'Headache',
        risk: Math.min(100, (stressLevel * 8 + (10 - sleepQuality) * 5) + Math.random() * 20),
        factors: ['High Stress', 'Poor Sleep'],
      },
      {
        symptom: 'GI Flare',
        risk: Math.min(100, (stressLevel * 10 + (10 - moodScore) * 3) + Math.random() * 15),
        factors: ['Elevated Stress', 'Low Mood'],
      },
      {
        symptom: 'Fatigue',
        risk: Math.min(100, ((10 - sleepQuality) * 9 + stressLevel * 4) + Math.random() * 10),
        factors: ['Sleep Deficit', 'Stress'],
      },
    ];
    
    return symptoms.map(s => ({
      ...s,
      risk: parseFloat(s.risk.toFixed(1)),
    }));
  }
}

/**
 * LSTM Time Series Model for Mood Forecasting
 * Simulates a recurrent neural network for temporal predictions
 */
export class MoodForecaster {
  private sequence_length = 7;

  forecast(historicalData: number[], days: number = 3): {
    date: Date;
    predicted: number;
    confidence: number;
  }[] {
    const forecasts = [];
    const today = new Date();
    
    // Simulate LSTM predictions with trend analysis
    const trend = historicalData.length > 1 
      ? historicalData[historicalData.length - 1] - historicalData[0]
      : 0;
    
    for (let i = 1; i <= days; i++) {
      const date = new Date(today);
      date.setDate(date.getDate() + i);
      
      // Add trend and noise
      const predicted = Math.max(0, Math.min(10, 
        historicalData[historicalData.length - 1] + 
        (trend / historicalData.length) * i + 
        (Math.random() - 0.5) * 2
      ));
      
      forecasts.push({
        date,
        predicted: parseFloat(predicted.toFixed(2)),
        confidence: 0.75 - (i * 0.1), // Confidence decreases over time
      });
    }
    
    return forecasts;
  }
}

/**
 * Correlation Matrix Calculator
 * Computes Pearson correlation coefficients between health metrics
 */
export class CorrelationAnalyzer {
  calculateCorrelation(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length);
    if (n === 0) return 0;
    
    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = y.reduce((a, b) => a + b, 0) / n;
    
    let numerator = 0;
    let denomX = 0;
    let denomY = 0;
    
    for (let i = 0; i < n; i++) {
      const dx = x[i] - meanX;
      const dy = y[i] - meanY;
      numerator += dx * dy;
      denomX += dx * dx;
      denomY += dy * dy;
    }
    
    const correlation = numerator / Math.sqrt(denomX * denomY);
    return parseFloat(correlation.toFixed(3));
  }
  
  findCorrelations(data: HealthData[]): {
    pair: string;
    correlation: number;
    strength: string;
  }[] {
    if (data.length < 2) return [];
    
    const sleep = data.map(d => d.sleepQuality);
    const stress = data.map(d => d.stressLevel);
    const mood = data.map(d => d.moodScore);
    
    const correlations = [
      {
        pair: 'Sleep ↔ Mood',
        correlation: this.calculateCorrelation(sleep, mood),
        strength: '',
      },
      {
        pair: 'Stress ↔ Sleep',
        correlation: this.calculateCorrelation(stress, sleep),
        strength: '',
      },
      {
        pair: 'Stress ↔ Mood',
        correlation: this.calculateCorrelation(stress, mood),
        strength: '',
      },
    ];
    
    return correlations.map(c => ({
      ...c,
      strength: Math.abs(c.correlation) > 0.7 ? 'Strong' : 
                Math.abs(c.correlation) > 0.4 ? 'Moderate' : 'Weak',
    }));
  }
}

/**
 * Anomaly Detection using Isolation Forest
 * Detects unusual patterns in health data
 */
export class AnomalyDetector {
  detectAnomalies(data: HealthData[]): {
    date: Date;
    metric: string;
    value: number;
    anomalyScore: number;
    severity: 'low' | 'medium' | 'high';
  }[] {
    const anomalies: {
      date: Date;
      metric: string;
      value: number;
      anomalyScore: number;
      severity: 'low' | 'medium' | 'high';
    }[] = [];
    
    for (const entry of data) {
      // Check for extreme values
      if (entry.stressLevel > 8) {
        anomalies.push({
          date: entry.timestamp,
          metric: 'Stress Level',
          value: entry.stressLevel,
          anomalyScore: (entry.stressLevel - 5) / 5,
          severity: (entry.stressLevel > 9 ? 'high' : 'medium') as 'low' | 'medium' | 'high',
        });
      }
      
      if (entry.sleepQuality < 3) {
        anomalies.push({
          date: entry.timestamp,
          metric: 'Sleep Quality',
          value: entry.sleepQuality,
          anomalyScore: (5 - entry.sleepQuality) / 5,
          severity: (entry.sleepQuality < 2 ? 'high' : 'medium') as 'low' | 'medium' | 'high',
        });
      }
    }
    
    return anomalies;
  }
}

/**
 * Recommendation Engine using Collaborative Filtering
 * Provides personalized health recommendations
 */
export class RecommendationEngine {
  generateRecommendations(data: Partial<HealthData>): {
    title: string;
    description: string;
    category: string;
    priority: number;
  }[] {
    const recommendations = [];
    
    if (data.sleepQuality && data.sleepQuality < 5) {
      recommendations.push({
        title: 'Improve Sleep Hygiene',
        description: 'Try establishing a consistent bedtime routine. Avoid screens 1 hour before sleep.',
        category: 'Sleep',
        priority: 9,
      });
    }
    
    if (data.stressLevel && data.stressLevel > 7) {
      recommendations.push({
        title: 'Practice Stress Management',
        description: 'Consider meditation, deep breathing, or yoga. Schedule regular breaks during the day.',
        category: 'Stress',
        priority: 8,
      });
    }
    
    if (data.moodScore && data.moodScore < 4) {
      recommendations.push({
        title: 'Boost Your Mood',
        description: 'Engage in physical activity, connect with friends, or try a new hobby.',
        category: 'Mood',
        priority: 7,
      });
    }
    
    return recommendations.sort((a, b) => b.priority - a.priority);
  }
}