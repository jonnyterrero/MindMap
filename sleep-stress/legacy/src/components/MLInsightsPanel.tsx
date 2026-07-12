"use client";

import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Brain, TrendingUp, Activity, AlertTriangle } from "lucide-react";
import { useEffect, useState } from "react";
import {
  SleepQualityPredictor,
  SymptomPredictor,
  MoodForecaster,
  RecommendationEngine,
} from "@/lib/ml-models";

export default function MLInsightsPanel() {
  const [predictions, setPredictions] = useState<any>(null);

  useEffect(() => {
    // Simulate ML model inference
    const sleepPredictor = new SleepQualityPredictor();
    const symptomPredictor = new SymptomPredictor();
    const moodForecaster = new MoodForecaster();
    const recommendationEngine = new RecommendationEngine();

    const mockData = {
      stressLevel: 6.5,
      moodScore: 7.2,
      sleepDuration: 7,
      sleepQuality: 6.8,
    };

    const sleepPrediction = sleepPredictor.predict(mockData);
    const symptomRisks = symptomPredictor.predictSymptomRisk(mockData);
    const moodForecast = moodForecaster.forecast([7, 7.5, 6.8, 7.2, 7.1, 6.9, 7.2]);
    const recommendations = recommendationEngine.generateRecommendations(mockData);

    setPredictions({
      sleepPrediction,
      symptomRisks,
      moodForecast,
      recommendations,
    });
  }, []);

  if (!predictions) {
    return (
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-5 h-5 text-purple-600 animate-pulse" />
          <h3 className="text-lg font-semibold">AI/ML Analysis</h3>
        </div>
        <p className="text-sm text-muted-foreground">Processing health data...</p>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Sleep Quality Prediction */}
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-5 h-5 text-purple-600" />
          <h3 className="text-lg font-semibold">Sleep Prediction (DNN)</h3>
        </div>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Expected Quality</span>
            <Badge variant="secondary" className="text-base font-bold">
              {predictions.sleepPrediction.prediction}/10
            </Badge>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Model Confidence</span>
            <span className="text-sm font-medium">
              {(predictions.sleepPrediction.confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div className="pt-2 border-t">
            <p className="text-xs text-muted-foreground">
              Model: {predictions.sleepPrediction.model}
            </p>
          </div>
        </div>
      </Card>

      {/* Symptom Risk Analysis */}
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="w-5 h-5 text-orange-600" />
          <h3 className="text-lg font-semibold">Symptom Risks (RF)</h3>
        </div>
        <div className="space-y-3">
          {predictions.symptomRisks.map((symptom: any, index: number) => (
            <div key={index} className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{symptom.symptom}</span>
                <span className="text-sm font-bold">{symptom.risk.toFixed(0)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all ${
                    symptom.risk > 70
                      ? "bg-red-500"
                      : symptom.risk > 40
                      ? "bg-orange-500"
                      : "bg-green-500"
                  }`}
                  style={{ width: `${symptom.risk}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Factors: {symptom.factors.join(", ")}
              </p>
            </div>
          ))}
        </div>
        <div className="pt-3 border-t mt-3">
          <p className="text-xs text-muted-foreground">
            Model: Random Forest Classifier (100 trees)
          </p>
        </div>
      </Card>

      {/* Mood Forecast */}
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold">3-Day Mood Forecast (LSTM)</h3>
        </div>
        <div className="space-y-3">
          {predictions.moodForecast.map((forecast: any, index: number) => (
            <div key={index} className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">
                  {forecast.date.toLocaleDateString("en-US", {
                    weekday: "short",
                    month: "short",
                    day: "numeric",
                  })}
                </p>
                <p className="text-xs text-muted-foreground">
                  Confidence: {(forecast.confidence * 100).toFixed(0)}%
                </p>
              </div>
              <Badge variant="outline" className="text-base">
                {forecast.predicted.toFixed(1)}/10
              </Badge>
            </div>
          ))}
        </div>
        <div className="pt-3 border-t mt-3">
          <p className="text-xs text-muted-foreground">
            Model: Long Short-Term Memory Network
          </p>
        </div>
      </Card>

      {/* AI Recommendations */}
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="w-5 h-5 text-green-600" />
          <h3 className="text-lg font-semibold">AI Recommendations</h3>
        </div>
        <div className="space-y-3">
          {predictions.recommendations.map((rec: any, index: number) => (
            <div key={index} className="p-3 rounded-lg bg-accent">
              <div className="flex items-start justify-between mb-1">
                <h4 className="text-sm font-semibold">{rec.title}</h4>
                <Badge variant="secondary" className="text-xs">
                  {rec.category}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">{rec.description}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}