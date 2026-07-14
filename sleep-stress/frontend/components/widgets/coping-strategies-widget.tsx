"use client"

import { Heart, Clock, Zap } from "lucide-react"

export function CopingStrategiesWidget() {
  const strategies = [
    {
      name: "Box Breathing",
      duration: 5,
      effectiveness: 4,
      category: "breathing",
      icon: <Heart className="w-5 h-5" />,
      color: "from-[var(--color-primary)] to-[var(--color-primary-dark)]",
    },
    {
      name: "Mindful Walking",
      duration: 20,
      effectiveness: 4,
      category: "physical",
      icon: <Zap className="w-5 h-5" />,
      color: "from-[var(--color-secondary)] to-[var(--color-secondary-dark)]",
    },
    {
      name: "Gratitude Journal",
      duration: 10,
      effectiveness: 3,
      category: "mindfulness",
      icon: <Heart className="w-5 h-5" />,
      color: "from-[var(--color-accent)] to-[var(--color-accent-dark)]",
    },
  ]

  return (
    <div className="glass rounded-[var(--radius-lg)] p-6">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-[var(--color-text)] mb-2">Coping Strategies</h3>
        <p className="text-sm text-[var(--color-text-secondary)]">Recommended for you today</p>
      </div>

      <div className="space-y-3">
        {strategies.map((strategy, index) => (
          <button
            key={index}
            className="w-full glass-hover rounded-[var(--radius)] p-4 text-left transition-all hover:scale-[1.02]"
          >
            <div className="flex items-center gap-4">
              <div
                className={`w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br ${strategy.color} flex items-center justify-center text-white`}
              >
                {strategy.icon}
              </div>
              <div className="flex-1">
                <p className="font-semibold text-[var(--color-text)] mb-1">{strategy.name}</p>
                <div className="flex items-center gap-3 text-xs text-[var(--color-text-secondary)]">
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    <span>{strategy.duration} min</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span>Effectiveness:</span>
                    <div className="flex gap-0.5">
                      {[...Array(5)].map((_, i) => (
                        <div
                          key={i}
                          className={`w-2 h-2 rounded-full ${i < strategy.effectiveness ? "bg-yellow-400" : "bg-gray-300"}`}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
