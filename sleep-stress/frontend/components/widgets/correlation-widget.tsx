"use client"

import { Activity } from "lucide-react"

export function CorrelationWidget() {
  const correlations = [
    { metric1: "Sleep", metric2: "Mood", value: 0.82, color: "from-green-400 to-emerald-500" },
    { metric1: "Stress", metric2: "GI Flare", value: 0.71, color: "from-orange-400 to-red-500" },
    { metric1: "Sleep", metric2: "Skin Health", value: -0.65, color: "from-blue-400 to-cyan-500" },
  ]

  return (
    <div className="glass rounded-[var(--radius-lg)] p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-accent)] to-[var(--color-accent-dark)] flex items-center justify-center">
          <Activity className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-[var(--color-text)]">Health Correlations</h3>
          <p className="text-sm text-[var(--color-text-secondary)]">Key patterns discovered</p>
        </div>
      </div>

      <div className="space-y-4">
        {correlations.map((corr, index) => (
          <div key={index} className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-[var(--color-text)]">{corr.metric1}</span>
                <span className="text-xs text-[var(--color-text-secondary)]">↔</span>
                <span className="text-sm font-medium text-[var(--color-text)]">{corr.metric2}</span>
              </div>
              <span className="text-sm font-bold text-[var(--color-text)]">
                {corr.value > 0 ? "+" : ""}
                {corr.value.toFixed(2)}
              </span>
            </div>
            <div className="h-2 bg-white/50 rounded-full overflow-hidden">
              <div
                className={`h-full bg-gradient-to-r ${corr.color} rounded-full`}
                style={{ width: `${Math.abs(corr.value) * 100}%` }}
              />
            </div>
            <p className="text-xs text-[var(--color-text-secondary)]">
              {corr.value > 0 ? "Positive" : "Negative"} correlation -{" "}
              {Math.abs(corr.value) > 0.7 ? "Strong" : "Moderate"} relationship
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
