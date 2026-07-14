"use client";

import { Card } from "@/components/ui/card";
import { Activity } from "lucide-react";
import { useState } from "react";

export default function SymptomTracking() {
  const [hasData] = useState(false);

  return (
    <Card className="p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-orange-100">
          <Activity className="w-5 h-5 text-orange-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">Symptom Tracking</h3>
          <p className="text-sm text-muted-foreground">Current & predicted</p>
        </div>
      </div>

      {!hasData ? (
        <div className="py-12 text-center">
          <p className="text-muted-foreground mb-2">No symptom data yet</p>
          <p className="text-sm text-muted-foreground">
            Start logging symptoms to track patterns
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {/* Symptom entries will be rendered here */}
          <p className="text-muted-foreground">Symptom entries</p>
        </div>
      )}
    </Card>
  );
}