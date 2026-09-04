"use client";

import { useState, useTransition } from "react";
import {
  createMedSchedule,
  toggleMedActive,
  updateMedSchedule,
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
} from "@/components/ui/card";
import { Plus, Trash2, Loader2, Pill, Clock, Pencil, Save, X } from "lucide-react";

type Schedule = Record<string, unknown>;

const FREQUENCIES = ["Daily", "Weekly", "Monthly", "As Needed", "Custom"];

export function MedicationsList({ schedules: initialSchedules }: { schedules: Schedule[] }) {
  const [isPending, startTransition] = useTransition();
  const [schedules, setSchedules] = useState(initialSchedules);
  const [showAdd, setShowAdd] = useState(false);
  const [name, setName] = useState("");
  const [dosage, setDosage] = useState("");
  const [doseMg, setDoseMg] = useState("");
  const [frequency, setFrequency] = useState("Daily");
  const [reminderTime, setReminderTime] = useState("08:00");
  const [notes, setNotes] = useState("");

  // Per-row inline edit state. When editingId is set, the row is replaced by
  // the same form the "Add" card uses, populated from the row's current values.
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editDosage, setEditDosage] = useState("");
  const [editDoseMg, setEditDoseMg] = useState("");
  const [editFrequency, setEditFrequency] = useState("Daily");
  const [editReminderTime, setEditReminderTime] = useState("08:00");
  const [editErr, setEditErr] = useState<string | null>(null);

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

    const optimistic: Schedule = {
      id: `temp-${Date.now()}`,
      ...payload,
      is_active: true,
      created_at: new Date().toISOString(),
    };
    setSchedules((prev) => [...prev, optimistic]);
    resetForm();

    startTransition(async () => {
      await createMedSchedule(payload);
    });
  }

  function handleToggle(id: string, current: boolean) {
    setSchedules((prev) =>
      prev.map((m) => (m.id === id ? { ...m, is_active: !current } : m))
    );
    startTransition(async () => {
      await toggleMedActive(id, !current);
    });
  }

  function handleDelete(id: string) {
    setSchedules((prev) => prev.filter((m) => m.id !== id));
    startTransition(async () => {
      await deleteMedSchedule(id);
    });
  }

  function startEdit(med: Schedule) {
    setEditingId(med.id as string);
    setEditName((med.name as string) ?? "");
    setEditDosage((med.dosage as string) ?? "");
    setEditDoseMg(med.dose_mg == null ? "" : String(med.dose_mg));
    setEditFrequency((med.frequency as string) ?? "Daily");
    setEditReminderTime((med.reminder_time as string) ?? "08:00");
    setEditErr(null);
  }

  function cancelEdit() {
    setEditingId(null);
    setEditErr(null);
  }

  function saveEdit() {
    if (!editingId) return;
    if (!editName.trim()) {
      setEditErr("Name can't be empty.");
      return;
    }
    const id = editingId;
    const patch: Partial<MedSchedulePayload> = {
      name: editName.trim(),
      dosage: editDosage.trim() || null,
      dose_mg: editDoseMg ? Number(editDoseMg) : null,
      frequency: editFrequency,
      reminder_time: editReminderTime || null,
    };

    setSchedules((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
    setEditingId(null);

    startTransition(async () => {
      const res = await updateMedSchedule(id, patch);
      if (res && "error" in res && res.error) {
        setEditErr(res.error);
        setEditingId(id);
      }
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
          {schedules.map((med) => {
            const medId = med.id as string;
            const isEditing = editingId === medId;
            const isTemp = medId.startsWith("temp-");
            return (
            <Card key={medId} className="glass-card">
              {isEditing ? (
                <CardContent className="space-y-4 py-4 px-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Name</Label>
                      <Input value={editName} onChange={(e) => setEditName(e.target.value)} />
                    </div>
                    <div className="space-y-2">
                      <Label>Dosage</Label>
                      <Input
                        placeholder="e.g. 50mg"
                        value={editDosage}
                        onChange={(e) => setEditDosage(e.target.value)}
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label>Dose (mg)</Label>
                      <Input
                        type="number"
                        value={editDoseMg}
                        onChange={(e) => setEditDoseMg(e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Frequency</Label>
                      <Select value={editFrequency} onValueChange={setEditFrequency}>
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
                        value={editReminderTime}
                        onChange={(e) => setEditReminderTime(e.target.value)}
                      />
                    </div>
                  </div>
                  {editErr && <p className="text-sm text-destructive">{editErr}</p>}
                  <div className="flex gap-2">
                    <Button onClick={saveEdit} disabled={isPending || !editName.trim()}>
                      {isPending ? <Loader2 className="animate-spin" /> : <Save className="h-4 w-4" />}
                      Save
                    </Button>
                    <Button variant="ghost" onClick={cancelEdit} disabled={isPending}>
                      <X className="h-4 w-4" /> Cancel
                    </Button>
                  </div>
                </CardContent>
              ) : (
              <CardContent className="flex items-center gap-4 py-4 px-4">
                <Switch
                  checked={med.is_active as boolean}
                  onCheckedChange={() =>
                    handleToggle(medId, med.is_active as boolean)
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
                    {(med.dosage as string) && (
                      <span className="text-muted-foreground font-normal ml-2">
                        {med.dosage as string}
                      </span>
                    )}
                  </p>
                  <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    <span>{med.frequency as string}</span>
                    {(med.reminder_time as string) && (
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {med.reminder_time as string}
                      </span>
                    )}
                  </div>
                </div>

                {!isTemp && (
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={() => startEdit(med)}
                    disabled={isPending}
                    aria-label={`Edit ${med.name as string}`}
                  >
                    <Pencil className="h-4 w-4" />
                  </Button>
                )}
                <Button
                  size="icon"
                  variant="ghost"
                  onClick={() => handleDelete(medId)}
                  disabled={isPending}
                  aria-label={`Delete ${med.name as string}`}
                >
                  <Trash2 className="h-4 w-4 text-destructive" />
                </Button>
              </CardContent>
              )}
            </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
