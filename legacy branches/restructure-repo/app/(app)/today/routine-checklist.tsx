"use client";

import { useState, useTransition } from "react";
import { toggleRoutineCompletion } from "./actions";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ListChecks } from "lucide-react";

type RoutineWithStatus = {
  id: string;
  name: string;
  completed: boolean;
};

export function RoutineChecklist({
  routines: initialRoutines,
}: {
  routines: RoutineWithStatus[];
}) {
  const [isPending, startTransition] = useTransition();
  const [routines, setRoutines] = useState(initialRoutines);

  const completedCount = routines.filter((r) => r.completed).length;

  function handleToggle(routineId: string, checked: boolean) {
    setRoutines((prev) =>
      prev.map((r) => (r.id === routineId ? { ...r, completed: checked } : r))
    );
    startTransition(async () => {
      await toggleRoutineCompletion(routineId, checked);
    });
  }

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ListChecks className="h-5 w-5 text-primary" /> Routines
        </CardTitle>
        <CardDescription>
          {completedCount}/{routines.length} completed today
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {routines.map((routine) => (
          <div key={routine.id} className="flex items-center gap-3">
            <Checkbox
              id={`routine-${routine.id}`}
              checked={routine.completed}
              onCheckedChange={(checked) =>
                handleToggle(routine.id, checked === true)
              }
              disabled={isPending}
            />
            <Label
              htmlFor={`routine-${routine.id}`}
              className={`cursor-pointer ${
                routine.completed ? "line-through text-muted-foreground" : ""
              }`}
            >
              {routine.name}
            </Label>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
