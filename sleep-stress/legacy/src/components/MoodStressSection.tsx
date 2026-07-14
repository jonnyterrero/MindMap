"use client";

import { Card } from "@/components/ui/card";
import { Heart } from "lucide-react";
import { useState } from "react";

export default function MoodStressSection() {
  const [hasData] = useState(false);

  return (
    <Card className="p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-pink-100">
          <Heart className="w-5 h-5 text-pink-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">Mood & Stress</h3>
          <p className="text-sm text-muted-foreground">Recent entries</p>
        </div>
      </div>

      {!hasData ? (
        <div className="py-12 text-center">
          <p className="text-muted-foreground mb-2">No mood data yet</p>
          <p className="text-sm text-muted-foreground">
            Start logging your mood to see patterns here
          </p>
        </div>
      ) : (
        <div className="h-64 flex items-center justify-center">
          {/* Chart will be rendered here when data exists */}
          <p className="text-muted-foreground">Mood data visualization</p>
        </div>
      )}
    </Card>
  );
}