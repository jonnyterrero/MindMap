"use client";

import { Card } from "@/components/ui/card";
import { ArrowUp, ArrowDown } from "lucide-react";
import { type ComponentType } from "react";

interface HealthMetricsCardProps {
  icon: ComponentType<{ className?: string }>;
  title: string;
  value: string;
  maxValue: string;
  trend?: number;
  iconColor: string;
  iconBg: string;
}

export default function HealthMetricsCard({
  icon: Icon,
  title,
  value,
  maxValue,
  trend,
  iconColor,
  iconBg,
}: HealthMetricsCardProps) {
  const isPositive = trend && trend > 0;
  const isNegative = trend && trend < 0;

  return (
    <Card className="p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <div className={`p-2 rounded-lg ${iconBg}`}>
              <Icon className={`w-4 h-4 ${iconColor}`} />
            </div>
            <span className="text-sm text-muted-foreground">{title}</span>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-3xl font-bold">{value}</span>
            <span className="text-lg text-muted-foreground">/{maxValue}</span>
          </div>
        </div>
        {trend !== undefined && (
          <div
            className={`flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium ${
              isPositive
                ? "bg-green-100 text-green-700"
                : isNegative
                ? "bg-red-100 text-red-700"
                : "bg-gray-100 text-gray-700"
            }`}
          >
            {isPositive ? (
              <ArrowUp className="w-3 h-3" />
            ) : isNegative ? (
              <ArrowDown className="w-3 h-3" />
            ) : null}
            <span>{Math.abs(trend)}</span>
          </div>
        )}
      </div>
    </Card>
  );
}