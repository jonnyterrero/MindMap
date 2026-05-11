"use client";

import { useState, useTransition } from "react";
import {
  createMedSchedule,
  toggleMedActive,
  deleteMedSchedule,
  type MedSchedulePayload,
} from "./actions";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Plus, Trash2, Loader2, Pill, Clock } from "lucide-react";

type Schedule = Record<string, unknown>;

const FREQUENCIES = ["Daily", "Weekly", "Monthly", "As Needed", "Custom"];

export function MedicationsList({ schedules }: { schedules: Schedule[] }) {
  const [isPending, startTransition] = useTransition();
  const [showAdd, setShowAdd] = useState(false);
  const [name, setName] = useState("");
  const [dosage, setDosage] = useState("");
  const [doseMg, setDoseMg] = useState("");
  const [frequency, setFrequency] = useState("Daily");
  const [reminderTime, setReminderTime] = useState("08:00");
  const [notes, setNotes] = useState("");

  function resetForm() {
    setName("");
    setDosage("");
    setDoseMg("");
    setFrequency("Daily");
    setReminderTime("08:00");
    setNotes("");
    setShowAdd(false);
  }

  function handleCreate() {
    if (!name.trim()) return;
    const payload: MedSchedulePayload = {
      name: name.trim(),
      dosage: dosage.trim() || null,
      dose_mg: doseMg ? Number(doseMg) : null,
      frequency,
      reminder_time: reminderTime || null,
      start_date: new Date().toISOString().split("T")[0],
      end_date: null,
      notes: notes.trim() || null,
    };
    startTransition(async () => {
      await createMedSchedule(payload);
      resetForm();
    });
  }

  function handleToggle(id: string, current: boolean) {
    startTransition(async () => {
      await toggleMedActive(id, !current);
    });
  }

  function handleDelete(id: string) {
    startTransition(async () => {
      await deleteMedSchedule(id);
    });
  }

  return (
    <div className="space-y-4">
      {!showAdd ? (
        <Button onClick={() => setShowAdd(true)}>
          <Plus className="h-4 w-4" /> Add Medication
        </Button>
      ) : (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-base">New Medication</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Name</Label>
                <Input
                  placeholder="e.g. Sertraline"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label>Dosage</Label>
                <Input
                  placeholder="e.g. 50mg"
                  value={dosage}
                  onChange={(e) => setDosage(e.target.value)}
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label>Dose (mg)</Label>
                <Input
                  type="number"
                  placeholder="50"
                  value={doseMg}
                  onChange={(e) => setDoseMg(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label>Frequency</Label>
                <Select value={frequency} onValueChange={setFrequency}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {FREQUENCIES.map((f) => (
                      <SelectItem key={f} value={f}>
                        {f}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Reminder time</Label>
                <Input
                  type="time"
                  value={reminderTime}
                  onChange={(e) => setReminderTime(e.target.value)}
                />
              </div>
            </div>

            <div className="flex gap-2">
              <Button onClick={handleCreate} disabled={isPending || !name.trim()}>
                {isPending ? <Loader2 className="animate-spin" /> : <Plus className="h-4 w-4" />}
                Add
              </Button>
              <Button variant="ghost" onClick={resetForm}>
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {schedules.length === 0 && !showAdd ? (
        <p className="text-center text-muted-foreground py-8">
          No medications yet. Add your first one above.
        </p>
      ) : (
        <div className="space-y-2">
          {schedules.map((med) => (
            <Card key={med.id as string} className="glass-card">
              <CardContent className="flex items-center gap-4 py-4 px-4">
                <Switch
                  checked={med.is_active as boolean}
                  onCheckedChange={() =>
                    handleToggle(med.id as string, med.is_active as boolean)
                  }
                  disabled={isPending}
                />

                <Pill className="h-4 w-4 text-primary shrink-0" />

                <div className="flex-1 min-w-0">
                  <p
                    className={`font-medium ${
                      !(med.is_active as boolean)
                        ? "line-through text-muted-foreground"
                        : ""
                    }`}
                  >
                    {med.name as string}
                    {med.dosage && (
                      <span className="text-muted-foreground font-normal ml-2">
                        {med.dosage as string}
                      </span>
                    )}
                  </p>
                  <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    <span>{med.frequency as string}</span>
                    {med.reminder_time && (
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {med.reminder_time as string}
                      </span>
                    )}
                  </div>
                </div>

                <Button
                  size="icon"
                  variant="ghost"
                  onClick={() => handleDelete(med.id as string)}
                  disabled={isPending}
                >
                  <Trash2 className="h-4 w-4 text-destructive" />
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
