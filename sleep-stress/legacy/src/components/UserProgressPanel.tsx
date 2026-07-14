"use client";

import { Card } from "@/components/ui/card";
import { Award, Flame, Trophy, Lightbulb } from "lucide-react";

export default function UserProgressPanel() {
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Your Progress</h3>
        <Award className="w-5 h-5 text-purple-600" />
      </div>

      <div className="space-y-4">
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Level 1</span>
            <span className="text-xs text-muted-foreground">0/600 XP</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div className="bg-purple-500 h-2 rounded-full w-0" />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 pt-4 border-t">
          <div className="text-center">
            <div className="flex items-center justify-center mb-1">
              <Flame className="w-4 h-4 text-orange-500" />
            </div>
            <p className="text-2xl font-bold">0</p>
            <p className="text-xs text-muted-foreground">days</p>
          </div>
          <div className="text-center">
            <div className="flex items-center justify-center mb-1">
              <Trophy className="w-4 h-4 text-yellow-500" />
            </div>
            <p className="text-2xl font-bold">0</p>
            <p className="text-xs text-muted-foreground">earned</p>
          </div>
        </div>

        <div className="pt-4 border-t">
          <div className="flex items-center gap-2 mb-3">
            <Lightbulb className="w-4 h-4 text-blue-600" />
            <h4 className="text-sm font-semibold">Start Your Journey</h4>
          </div>
          <p className="text-xs text-muted-foreground">
            Log data to earn achievements
          </p>
        </div>
      </div>
    </Card>
  );
}