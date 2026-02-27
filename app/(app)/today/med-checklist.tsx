"use client";

import { useTransition } from "react";
import { logMedAdherence } from "@/app/(app)/medications/actions";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Pill, Clock } from "lucide-react";

type MedWithStatus = {
  id: string;
  name: string;
  dosage: string | null;
  reminder_time: string | null;
  was_taken: boolean;
  was_skipped: boolean;
};

export function MedChecklist({ meds }: { meds: MedWithStatus[] }) {
  const [isPending, startTransition] = useTransition();

  const takenCount = meds.filter((m) => m.was_taken).length;

  function handleToggle(scheduleId: string, currentlyTaken: boolean) {
    startTransition(async () => {
      await logMedAdherence(scheduleId, !currentlyTaken, false);
    });
  }

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Pill className="h-5 w-5 text-primary" /> Medications
        </CardTitle>
        <CardDescription>
          {takenCount}/{meds.length} taken today
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {meds.map((med) => (
          <div key={med.id} className="flex items-center gap-3">
            <Checkbox
              id={`med-${med.id}`}
              checked={med.was_taken}
              onCheckedChange={() => handleToggle(med.id, med.was_taken)}
              disabled={isPending}
            />
            <Label
              htmlFor={`med-${med.id}`}
              className={`cursor-pointer flex-1 ${
                med.was_taken ? "line-through text-muted-foreground" : ""
              }`}
            >
              {med.name}
              {med.dosage && (
                <span className="text-muted-foreground ml-1">{med.dosage}</span>
              )}
            </Label>
            {med.reminder_time && (
              <span className="text-xs text-muted-foreground flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {med.reminder_time}
              </span>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
