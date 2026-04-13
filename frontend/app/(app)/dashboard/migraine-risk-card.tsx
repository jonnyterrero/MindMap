"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { AlertTriangle, Shield, ShieldAlert } from "lucide-react";

type MigraineRisk = {
  score: number;
  factors: string[];
  entries: number;
};

function getRiskLevel(score: number) {
  if (score >= 60) return { label: "High", color: "text-red-500", icon: ShieldAlert };
  if (score >= 30) return { label: "Moderate", color: "text-yellow-500", icon: AlertTriangle };
  return { label: "Low", color: "text-green-500", icon: Shield };
}

export function MigraineRiskCard({ risk }: { risk: MigraineRisk }) {
  const level = getRiskLevel(risk.score);
  const Icon = level.icon;

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Icon className={`h-5 w-5 ${level.color}`} />
          Migraine Risk Today
        </CardTitle>
        <CardDescription>
          Based on your last 7 days of data ({risk.entries} entries)
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className={`font-medium ${level.color}`}>{level.label} Risk</span>
            <span className="font-mono">{risk.score}/100</span>
          </div>
          <Progress value={risk.score} className="h-2" />
        </div>

        {risk.factors.length > 0 && (
          <div className="space-y-1">
            <p className="text-sm font-medium">Contributing factors:</p>
            <ul className="text-sm text-muted-foreground space-y-0.5">
              {risk.factors.map((factor, i) => (
                <li key={i} className="flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground shrink-0" />
                  {factor}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
