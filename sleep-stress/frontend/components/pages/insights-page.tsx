"use client"

import { Sparkles, TrendingUp, AlertCircle, Lightbulb, Brain, Target } from "lucide-react"

export function InsightsPage() {
  const insights = [
    {
      id: 1,
      type: "positive",
      category: "sleep",
      icon: <TrendingUp className="w-5 h-5" />,
      title: "Sleep Quality Improving",
      description: "Your sleep quality has improved from 6.8 to 8.2 over the past week. Keep up the great work!",
      recommendations: [
        "Try going to bed 30 minutes earlier",
        "Create a consistent bedtime routine",
        "Avoid screens 1 hour before bed",
        "Keep your bedroom cool and dark",
      ],
      priority: "high",
      confidence: 0.9,
      color: "from-green-400 to-emerald-500",
    },
    {
      id: 2,
      type: "warning",
      category: "stress",
      icon: <AlertCircle className="w-5 h-5" />,
      title: "High Stress Alert",
      description: "Your stress levels have been above 7 for 3 consecutive days. This may impact your overall health.",
      recommendations: [
        "Practice deep breathing exercises",
        "Try progressive muscle relaxation",
        "Consider stress management therapy",
        "Identify and address stress sources",
      ],
      priority: "high",
      confidence: 0.85,
      color: "from-orange-400 to-red-500",
    },
    {
      id: 3,
      type: "correlation",
      category: "health",
      icon: <Brain className="w-5 h-5" />,
      title: "Sleep-Mood Connection",
      description:
        "Strong positive correlation detected between sleep quality and mood scores. Better sleep leads to better mood.",
      recommendations: [
        "Prioritize sleep quality to improve mood",
        "Track sleep patterns more consistently",
        "Consider sleep hygiene improvements",
      ],
      priority: "medium",
      confidence: 0.82,
      color: "from-blue-400 to-cyan-500",
    },
    {
      id: 4,
      type: "tip",
      category: "lifestyle",
      icon: <Lightbulb className="w-5 h-5" />,
      title: "Weekend Sleep Catch-up",
      description:
        "You sleep 1.5 hours more on weekends, suggesting weekday sleep debt. Try to maintain consistent sleep schedule.",
      recommendations: [
        "Try to get more consistent sleep during weekdays",
        "Consider adjusting weekday bedtime",
        "Avoid oversleeping on weekends",
      ],
      priority: "medium",
      confidence: 0.78,
      color: "from-purple-400 to-purple-600",
    },
    {
      id: 5,
      type: "achievement",
      category: "progress",
      icon: <Target className="w-5 h-5" />,
      title: "Mood Improvement Milestone",
      description: "Your average mood score increased by 2 points over the last 7 days. Excellent progress!",
      recommendations: ["Continue current positive habits", "Share your success strategies", "Set new wellness goals"],
      priority: "low",
      confidence: 0.95,
      color: "from-yellow-400 to-orange-500",
    },
  ]

  const predictions = [
    {
      metric: "GI Flare",
      current: 3,
      predicted: 2,
      probability: 0.25,
      risk: "low",
      factors: ["Good sleep quality", "Low stress levels", "Consistent routine"],
    },
    {
      metric: "Mood Score",
      current: 7.8,
      predicted: 8.2,
      probability: 0.75,
      risk: "positive",
      factors: ["Improving sleep", "Reduced stress", "Active coping strategies"],
    },
    {
      metric: "Skin Flare",
      current: 4,
      predicted: 6,
      probability: 0.65,
      risk: "medium",
      factors: ["Increased stress", "Weather changes", "Sleep disruption"],
    },
  ]

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "low":
        return "from-green-400 to-emerald-500"
      case "medium":
        return "from-yellow-400 to-orange-500"
      case "high":
        return "from-orange-500 to-red-600"
      case "positive":
        return "from-blue-400 to-cyan-500"
      default:
        return "from-gray-400 to-gray-500"
    }
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="glass rounded-[var(--radius-lg)] p-6 md:p-8">
        <div className="flex items-center gap-4 mb-2">
          <div className="w-14 h-14 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-secondary)] flex items-center justify-center">
            <Sparkles className="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 className="text-3xl md:text-4xl font-bold gradient-text">AI-Powered Insights</h1>
            <p className="text-[var(--color-text-secondary)]">Personalized health recommendations based on your data</p>
          </div>
        </div>
      </div>

      {/* Predictions Section */}
      <div className="glass rounded-[var(--radius-lg)] p-6">
        <h2 className="text-2xl font-bold text-[var(--color-text)] mb-6">Tomorrow's Predictions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {predictions.map((pred, index) => (
            <div key={index} className="glass-hover rounded-[var(--radius)] p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-[var(--color-text)]">{pred.metric}</h3>
                <span
                  className={`px-3 py-1 rounded-full text-xs font-medium text-white bg-gradient-to-r ${getRiskColor(pred.risk)}`}
                >
                  {pred.risk === "positive" ? "improving" : `${pred.risk} risk`}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <p className="text-xs text-[var(--color-text-secondary)] mb-1">Current</p>
                  <p className="text-2xl font-bold text-[var(--color-text)]">{pred.current}</p>
                </div>
                <div>
                  <p className="text-xs text-[var(--color-text-secondary)] mb-1">Predicted</p>
                  <p className="text-2xl font-bold text-[var(--color-text)]">{pred.predicted}</p>
                </div>
              </div>

              <div className="mb-3">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-[var(--color-text-secondary)]">Confidence</span>
                  <span className="font-medium text-[var(--color-text)]">{(pred.probability * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 bg-white/50 rounded-full overflow-hidden">
                  <div
                    className={`h-full bg-gradient-to-r ${getRiskColor(pred.risk)} rounded-full`}
                    style={{ width: `${pred.probability * 100}%` }}
                  />
                </div>
              </div>

              <div className="space-y-1">
                <p className="text-xs font-semibold text-[var(--color-text-secondary)]">Key Factors:</p>
                {pred.factors.map((factor, i) => (
                  <p key={i} className="text-xs text-[var(--color-text-muted)]">
                    • {factor}
                  </p>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Insights Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {insights.map((insight) => (
          <div
            key={insight.id}
            className="glass rounded-[var(--radius-lg)] p-6 border-l-4"
            style={{
              borderLeftColor:
                insight.type === "positive" || insight.type === "achievement"
                  ? "var(--color-success)"
                  : insight.type === "warning"
                    ? "var(--color-warning)"
                    : "var(--color-primary)",
            }}
          >
            {/* Header */}
            <div className="flex items-start gap-4 mb-4">
              <div
                className={`w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br ${insight.color} flex items-center justify-center text-white flex-shrink-0`}
              >
                {insight.icon}
              </div>
              <div className="flex-1">
                <div className="flex items-start justify-between gap-2 mb-2">
                  <h3 className="text-xl font-bold text-[var(--color-text)]">{insight.title}</h3>
                  <span className="text-xs px-2 py-1 bg-white/60 rounded-full text-[var(--color-text-secondary)] whitespace-nowrap">
                    {insight.category}
                  </span>
                </div>
                <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{insight.description}</p>
              </div>
            </div>

            {/* Confidence & Priority */}
            <div className="flex items-center gap-4 mb-4 pb-4 border-b border-[var(--color-border)]">
              <div className="flex items-center gap-2">
                <span className="text-xs text-[var(--color-text-secondary)]">Confidence:</span>
                <span className="text-sm font-bold text-[var(--color-text)]">
                  {(insight.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-[var(--color-text-secondary)]">Priority:</span>
                <span
                  className={`text-xs font-medium px-2 py-1 rounded-full ${
                    insight.priority === "high"
                      ? "bg-red-100 text-red-700"
                      : insight.priority === "medium"
                        ? "bg-yellow-100 text-yellow-700"
                        : "bg-green-100 text-green-700"
                  }`}
                >
                  {insight.priority}
                </span>
              </div>
            </div>

            {/* Recommendations */}
            <div>
              <p className="text-sm font-semibold text-[var(--color-text)] mb-3">Recommendations:</p>
              <ul className="space-y-2">
                {insight.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start gap-2 text-sm text-[var(--color-text-secondary)]">
                    <span className="text-[var(--color-primary)] mt-1">•</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
