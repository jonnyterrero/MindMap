"use client";

import { useEffect, useState, useTransition } from "react";
import {
  createReminder,
  deleteReminder,
  setReminderActive,
  type ReminderType,
} from "./reminder-actions";
import {
  canScheduleDeviceNotifications,
  ensureNotificationPermission,
  syncDeviceNotifications,
  type ReminderRow,
} from "@/lib/notifications";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Bell, Plus, Trash2, Loader2, Smartphone } from "lucide-react";

const TYPE_OPTIONS: { value: ReminderType; label: string; defaultTitle: string }[] = [
  { value: "Check-in", label: "Daily check-in", defaultTitle: "Daily check-in" },
  { value: "Medication", label: "Medication", defaultTitle: "Medication reminder" },
  { value: "Journal", label: "Journal", defaultTitle: "Journal prompt" },
  { value: "Routine", label: "Routine", defaultTitle: "Routine reminder" },
  { value: "Therapy", label: "Therapy", defaultTitle: "Therapy reminder" },
  { value: "Custom", label: "Custom", defaultTitle: "" },
];

const DAY_LABELS = ["S", "M", "T", "W", "T", "F", "S"]; // index 0 = Sunday → weekday 1

export function ReminderSettings({ reminders: initial }: { reminders: ReminderRow[] }) {
  const [reminders, setReminders] = useState(initial);
  const [isPending, startTransition] = useTransition();
  const [err, setErr] = useState<string | null>(null);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const [isNative, setIsNative] = useState(false);

  // Capacitor detection must run client-side after hydration.
  useEffect(() => {
    setIsNative(canScheduleDeviceNotifications());
  }, []);

  // Keep the device schedule in step with the list whenever it changes.
  async function applyResult(result: { error: string } | { reminders: ReminderRow[] }) {
    if ("error" in result) {
      setErr(result.error);
      return;
    }
    setErr(null);
    setReminders(result.reminders);
    if (canScheduleDeviceNotifications()) {
      const granted = await ensureNotificationPermission();
      setPermissionDenied(!granted);
      if (granted) await syncDeviceNotifications(result.reminders);
    }
  }

  // Add form state
  const [showAdd, setShowAdd] = useState(false);
  const [type, setType] = useState<ReminderType>("Check-in");
  const [title, setTitle] = useState("Daily check-in");
  const [time, setTime] = useState("20:00");
  const [days, setDays] = useState<number[]>([]); // empty = every day

  function handleTypeChange(next: ReminderType) {
    setType(next);
    const opt = TYPE_OPTIONS.find((o) => o.value === next);
    if (opt && (title === "" || TYPE_OPTIONS.some((o) => o.defaultTitle === title))) {
      setTitle(opt.defaultTitle);
    }
  }

  function toggleDay(weekday: number) {
    setDays((prev) =>
      prev.includes(weekday) ? prev.filter((d) => d !== weekday) : [...prev, weekday].sort(),
    );
  }

  function handleAdd() {
    startTransition(async () => {
      const result = await createReminder({
        title,
        reminderType: type,
        reminderTime: time,
        daysOfWeek: days.length > 0 ? days : null,
      });
      await applyResult(result);
      if (!("error" in result)) {
        setShowAdd(false);
        setType("Check-in");
        setTitle("Daily check-in");
        setTime("20:00");
        setDays([]);
      }
    });
  }

  function handleToggle(id: string, next: boolean) {
    // Optimistic flip; applyResult replaces with server truth.
    setReminders((prev) => prev.map((r) => (r.id === id ? { ...r, is_active: next } : r)));
    startTransition(async () => {
      await applyResult(await setReminderActive(id, next));
    });
  }

  function handleDelete(id: string) {
    setReminders((prev) => prev.filter((r) => r.id !== id));
    startTransition(async () => {
      await applyResult(await deleteReminder(id));
    });
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <Bell className="h-5 w-5 text-primary" /> Reminders
        </CardTitle>
        <CardDescription>
          Daily nudges for check-ins, meds, and journaling so tracking becomes a habit.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {!isNative && (
          <div className="flex items-start gap-2 rounded-md bg-muted/60 p-3 text-sm text-muted-foreground">
            <Smartphone className="h-4 w-4 mt-0.5 shrink-0" />
            <p>
              Reminders ring on your phone. Set them up here — they&apos;ll schedule
              automatically when you open the MindMap mobile app.
            </p>
          </div>
        )}
        {permissionDenied && (
          <p className="text-sm text-destructive">
            Notifications are blocked for MindMap. Enable them in your device settings,
            then toggle a reminder to re-sync.
          </p>
        )}

        {reminders.length === 0 && !showAdd ? (
          <div className="rounded-md border border-dashed p-4 text-center space-y-2">
            <p className="text-sm text-muted-foreground">No reminders yet.</p>
            <Button
              variant="outline"
              size="sm"
              disabled={isPending}
              onClick={() =>
                startTransition(async () => {
                  await applyResult(
                    await createReminder({
                      title: "Daily check-in",
                      reminderType: "Check-in",
                      reminderTime: "20:00",
                      daysOfWeek: null,
                    }),
                  );
                })
              }
            >
              {isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
              Daily check-in at 8:00 PM
            </Button>
          </div>
        ) : (
          <ul className="space-y-2">
            {reminders.map((r) => (
              <li
                key={r.id}
                className="flex items-center justify-between gap-3 rounded-md border p-3"
              >
                <div className="min-w-0">
                  <p className="text-sm font-medium truncate">{r.title}</p>
                  <p className="text-xs text-muted-foreground">
                    {r.reminder_type} · {formatTime(r.reminder_time)} ·{" "}
                    {formatDays(r.days_of_week)}
                  </p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <Switch
                    checked={r.is_active}
                    onCheckedChange={(next) => handleToggle(r.id, next)}
                    disabled={isPending}
                    aria-label={`Toggle ${r.title}`}
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-destructive"
                    onClick={() => handleDelete(r.id)}
                    disabled={isPending}
                    aria-label={`Delete ${r.title}`}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </li>
            ))}
          </ul>
        )}

        {showAdd ? (
          <div className="space-y-3 rounded-md border p-3">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label htmlFor="reminder-type">Type</Label>
                <Select value={type} onValueChange={(v) => handleTypeChange(v as ReminderType)}>
                  <SelectTrigger id="reminder-type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TYPE_OPTIONS.map((o) => (
                      <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="reminder-time">Time</Label>
                <Input
                  id="reminder-time"
                  type="time"
                  value={time}
                  onChange={(e) => setTime(e.target.value)}
                />
              </div>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="reminder-title">Name</Label>
              <Input
                id="reminder-title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. Evening check-in"
                maxLength={80}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Days (leave empty for every day)</Label>
              <div className="flex gap-1.5">
                {DAY_LABELS.map((label, i) => {
                  const weekday = i + 1; // 1=Sunday … 7=Saturday
                  const selected = days.includes(weekday);
                  return (
                    <button
                      key={weekday}
                      type="button"
                      onClick={() => toggleDay(weekday)}
                      aria-pressed={selected}
                      className={`h-8 w-8 rounded-full text-xs font-medium transition-colors ${
                        selected
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted text-muted-foreground hover:bg-muted/80"
                      }`}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="flex gap-2">
              <Button size="sm" onClick={handleAdd} disabled={isPending || !title.trim()}>
                {isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                Add reminder
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setShowAdd(false)} disabled={isPending}>
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          reminders.length > 0 && (
            <Button variant="outline" size="sm" onClick={() => setShowAdd(true)}>
              <Plus className="h-4 w-4" /> Add reminder
            </Button>
          )
        )}

        {err && <p className="text-sm text-destructive">{err}</p>}
      </CardContent>
    </Card>
  );
}

function formatTime(t: string): string {
  const [hh, mm] = t.split(":").map(Number);
  const ampm = hh >= 12 ? "PM" : "AM";
  const hour12 = hh % 12 === 0 ? 12 : hh % 12;
  return `${hour12}:${String(mm).padStart(2, "0")} ${ampm}`;
}

function formatDays(days: number[] | null): string {
  if (!days || days.length === 0 || days.length === 7) return "Every day";
  const names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  return days.map((d) => names[d - 1]).join(", ");
}
