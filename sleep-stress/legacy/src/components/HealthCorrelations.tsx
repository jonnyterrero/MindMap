"use client";

import { Card } from "@/components/ui/card";
import { TrendingUp } from "lucide-react";

interface Correlation {
  label: string;
  correlation: number;
  description: string;
  color: string;
}

const correlations: Correlation[] = [
  {
    label: "Sleep ↔ Mood",
    correlation: 0.82,
    description: "Positive correlation - Strong relationship",
    color: "bg-green-500",
  },
  {
    label: "Stress ↔ GI Flare",
    correlation: 0.71,
    description: "Positive correlation - Strong relationship",
    color: "bg-orange-500",
  },
  {
    label: "Sleep ↔ Headaches",
    correlation: -0.65,
    description: "Negative correlation - Moderate relationship",
    color: "bg-cyan-500",
  },
];

export default function HealthCorrelations() {
  return (
    <Card className="p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-blue-100">
          <TrendingUp className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">Health Correlations</h3>
          <p className="text-sm text-muted-foreground">Key patterns discovered</p>
        </div>
      </div>

      <div className="space-y-4">
        {correlations.map((item, index) => (
          <div key={index} className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">{item.label}</span>
              <span className="text-sm font-bold">+{item.correlation.toFixed(2)}</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`${item.color} h-2 rounded-full transition-all`}
                style={{ width: `${Math.abs(item.correlation) * 100}%` }}
              />
            </div>
            <p className="text-xs text-muted-foreground">{item.description}</p>
          </div>
        ))}
      </div>
    </Card>
  );
}