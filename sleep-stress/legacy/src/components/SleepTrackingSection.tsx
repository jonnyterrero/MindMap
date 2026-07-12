"use client";

import { Card } from "@/components/ui/card";
import { Moon } from "lucide-react";
import { useState } from "react";

export default function SleepTrackingSection() {
  const [hasData] = useState(false);

  return (
    <Card className="p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-blue-100">
          <Moon className="w-5 h-5 text-blue-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">Sleep Tracking</h3>
          <p className="text-sm text-muted-foreground">Last 7 days</p>
        </div>
      </div>

      {!hasData ? (
        <div className="py-12 text-center">
          <p className="text-muted-foreground mb-2">No sleep data yet</p>
          <p className="text-sm text-muted-foreground">
            Start logging your sleep to see insights here
          </p>
        </div>
      ) : (
        <div className="h-64 flex items-center justify-center">
          {/* Chart will be rendered here when data exists */}
          <p className="text-muted-foreground">Sleep data visualization</p>
        </div>
      )}
    </Card>
  );
}