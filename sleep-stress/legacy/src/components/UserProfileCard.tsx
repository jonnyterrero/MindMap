"use client";

import { Card } from "@/components/ui/card";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Activity, Calculator } from "lucide-react";

export default function UserProfileCard() {
  return (
    <Card className="p-6">
      <div className="flex items-center gap-3 mb-6">
        <Avatar className="w-12 h-12 bg-purple-100">
          <AvatarFallback className="bg-purple-500 text-white font-semibold">
            G
          </AvatarFallback>
        </Avatar>
        <div>
          <h3 className="font-semibold">Guest User</h3>
          <p className="text-xs text-muted-foreground">Age not set</p>
        </div>
      </div>

      <div className="space-y-3">
        <div>
          <h4 className="text-sm font-semibold mb-2">Profile Completeness</h4>
          <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
            <div className="bg-purple-500 h-2 rounded-full w-0" />
          </div>
          <p className="text-xs text-muted-foreground">0%</p>
        </div>

        <div className="pt-3 border-t space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-muted-foreground" />
              <span className="text-xs">BMI</span>
            </div>
            <span className="text-xs font-medium">-- Not calculated</span>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Calculator className="w-4 h-4 text-muted-foreground" />
              <span className="text-xs">Conditions</span>
            </div>
            <span className="text-xs font-medium">0 tracked</span>
          </div>
        </div>

        <div className="pt-3 border-t">
          <p className="text-xs text-muted-foreground">
            No conditions added yet
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Update your profile to add conditions
          </p>
        </div>
      </div>
    </Card>
  );
}