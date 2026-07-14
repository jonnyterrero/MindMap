"use client";

import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Sparkles, TrendingUp, AlertCircle, ChevronRight } from "lucide-react";

interface Insight {
  id: string;
  type: "positive" | "warning" | "info";
  title: string;
  description: string;
  icon: typeof TrendingUp;
  iconColor: string;
  iconBg: string;
}

const insights: Insight[] = [
  {
    id: "1",
    type: "positive",
    title: "Sleep Improving",
    description: "Your sleep quality increased by 15% this week",
    icon: TrendingUp,
    iconColor: "text-green-600",
    iconBg: "bg-green-100",
  },
  {
    id: "2",
    type: "warning",
    title: "High Stress Alert",
    description: "Stress levels above 7 for 3 consecutive days",
    icon: AlertCircle,
    iconColor: "text-orange-600",
    iconBg: "bg-orange-100",
  },
];

export default function AIInsightsPanel() {
  return (
    <Card className="p-6">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-5 h-5 text-purple-600" />
        <h3 className="text-lg font-semibold">AI Insights</h3>
      </div>

      <div className="space-y-3">
        {insights.map((insight) => (
          <div
            key={insight.id}
            className="p-4 rounded-lg border border-border hover:bg-accent transition-colors"
          >
            <div className="flex items-start gap-3">
              <div className={`p-2 rounded-lg ${insight.iconBg}`}>
                <insight.icon className={`w-4 h-4 ${insight.iconColor}`} />
              </div>
              <div className="flex-1">
                <h4 className="font-medium text-sm mb-1">{insight.title}</h4>
                <p className="text-xs text-muted-foreground">{insight.description}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <Button variant="ghost" className="w-full mt-4 text-sm">
        View All Insights
        <ChevronRight className="w-4 h-4 ml-1" />
      </Button>
    </Card>
  );
}