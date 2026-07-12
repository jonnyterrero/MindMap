"use client"

import { Link2, Brain, Activity, Heart } from "lucide-react"

export function CrossAppWidget() {
  const apps = [
    {
      name: "MindMap",
      icon: <Brain className="w-5 h-5" />,
      status: "connected",
      lastSync: "2 min ago",
      color: "from-purple-400 to-purple-600",
    },
    {
      name: "SkinTrack+",
      icon: <Activity className="w-5 h-5" />,
      status: "connected",
      lastSync: "5 min ago",
      color: "from-pink-400 to-pink-600",
    },
    {
      name: "GastroGuard",
      icon: <Heart className="w-5 h-5" />,
      status: "connected",
      lastSync: "1 min ago",
      color: "from-blue-400 to-blue-600",
    },
  ]

  return (
    <div className="glass rounded-[var(--radius-lg)] p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-[var(--radius)] bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-accent)] flex items-center justify-center">
          <Link2 className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-[var(--color-text)]">Connected Apps</h3>
          <p className="text-sm text-[var(--color-text-secondary)]">Unified health tracking</p>
        </div>
      </div>

      <div className="space-y-3">
        {apps.map((app, index) => (
          <div key={index} className="glass-hover rounded-[var(--radius)] p-4">
            <div className="flex items-center gap-3">
              <div
                className={`w-10 h-10 rounded-[var(--radius-sm)] bg-gradient-to-br ${app.color} flex items-center justify-center text-white`}
              >
                {app.icon}
              </div>
              <div className="flex-1">
                <p className="font-semibold text-[var(--color-text)]">{app.name}</p>
                <p className="text-xs text-[var(--color-text-secondary)]">Synced {app.lastSync}</p>
              </div>
              <div className="w-2 h-2 rounded-full bg-green-500" />
            </div>
          </div>
        ))}
      </div>

      <button className="w-full mt-4 py-3 bg-gradient-to-r from-[var(--color-glass-purple)] to-[var(--color-glass-blue)] rounded-[var(--radius)] font-medium text-[var(--color-text)] hover:opacity-90 transition-opacity">
        View Unified Dashboard
      </button>
    </div>
  )
}
