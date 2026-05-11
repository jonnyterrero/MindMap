"use client";

import { useState, useTransition } from "react";
import { createTherapySession, deleteTherapySession, type TherapyPayload } from "./actions";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Plus, Trash2, Loader2, Heart, ArrowRight } from "lucide-react";
import { format, parseISO } from "date-fns";

type Session = Record<string, unknown>;

const SESSION_TYPES = ["Individual", "Group", "Couples", "Family", "Other"];
const MOOD_LABELS: Record<number, string> = {
  [-3]: "Very Low", [-2]: "Low", [-1]: "Slightly Low",
  0: "Neutral", 1: "Slightly High", 2: "High", 3: "Very High",
};

export function TherapyList({ sessions: initialSessions }: { sessions: Session[] }) {
  const [isPending, startTransition] = useTransition();
  const [sessions, setSessions] = useState(initialSessions);
  const [showNew, setShowNew] = useState(false);
  const [sessionDate, setSessionDate] = useState(new Date().toISOString().split("T")[0]);
  const [sessionTime, setSessionTime] = useState("");
  const [duration, setDuration] = useState("50");
  const [therapistName, setTherapistName] = useState("");
  const [sessionType, setSessionType] = useState("Individual");
  const [moodBefore, setMoodBefore] = useState(0);
  const [moodAfter, setMoodAfter] = useState(0);
  const [notes, setNotes] = useState("");
  const [homework, setHomework] = useState("");
  const [nextDate, setNextDate] = useState("");

  function resetForm() {
    setSessionDate(new Date().toISOString().split("T")[0]);
    setSessionTime(""); setDuration("50"); setTherapistName("");
    setSessionType("Individual"); setMoodBefore(0); setMoodAfter(0);
    setNotes(""); setHomework(""); setNextDate("");
    setShowNew(false);
  }

  function handleCreate() {
    const payload: TherapyPayload = {
      session_date: sessionDate,
      session_time: sessionTime || null,
      duration_minutes: duration ? Number(duration) : null,
      therapist_name: therapistName.trim() || null,
      session_type: sessionType,
      notes: notes.trim() || null,
      mood_before: moodBefore,
      mood_after: moodAfter,
      topics_discussed: [],
      homework_assigned: homework.trim() || null,
      next_session_date: nextDate || null,
    };

    const optimistic: Session = {
      id: `temp-${Date.now()}`,
      ...payload,
      created_at: new Date().toISOString(),
    };
    setSessions((prev) => [optimistic, ...prev]);
    resetForm();

    startTransition(async () => {
      await createTherapySession(payload);
    });
  }

  function handleDelete(id: string) {
    setSessions((prev) => prev.filter((s) => s.id !== id));
    startTransition(async () => {
      await deleteTherapySession(id);
    });
  }

  return (
    <div className="space-y-4">
      {!showNew ? (
        <Button onClick={() => setShowNew(true)}>
          <Plus className="h-4 w-4" /> Log Session
        </Button>
      ) : (
        <Card className="glass-card">
          <CardHeader><CardTitle className="text-base">New Session</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label>Date</Label>
                <Input type="date" value={sessionDate} onChange={(e) => setSessionDate(e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Time</Label>
                <Input type="time" value={sessionTime} onChange={(e) => setSessionTime(e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Duration (min)</Label>
                <Input type="number" value={duration} onChange={(e) => setDuration(e.target.value)} />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Therapist</Label>
                <Input placeholder="Name" value={therapistName} onChange={(e) => setTherapistName(e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Type</Label>
                <Select value={sessionType} onValueChange={setSessionType}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {SESSION_TYPES.map((t) => (<SelectItem key={t} value={t}>{t}</SelectItem>))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex justify-between"><Label>Mood before</Label><span className="text-sm">{MOOD_LABELS[moodBefore]}</span></div>
                <Slider min={-3} max={3} step={1} value={[moodBefore]} onValueChange={([v]) => setMoodBefore(v)} />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between"><Label>Mood after</Label><span className="text-sm">{MOOD_LABELS[moodAfter]}</span></div>
                <Slider min={-3} max={3} step={1} value={[moodAfter]} onValueChange={([v]) => setMoodAfter(v)} />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Session notes</Label>
              <Textarea placeholder="Key takeaways..." value={notes} onChange={(e) => setNotes(e.target.value)} rows={3} />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Homework</Label>
                <Input placeholder="Assigned tasks" value={homework} onChange={(e) => setHomework(e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Next session</Label>
                <Input type="date" value={nextDate} onChange={(e) => setNextDate(e.target.value)} />
              </div>
            </div>
            <div className="flex gap-2">
              <Button onClick={handleCreate} disabled={isPending}>
                {isPending ? <Loader2 className="animate-spin" /> : <Plus className="h-4 w-4" />} Save
              </Button>
              <Button variant="ghost" onClick={resetForm}>Cancel</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {sessions.length === 0 && !showNew && (
        <p className="text-center text-muted-foreground py-8">No sessions logged yet.</p>
      )}

      {sessions.map((s) => (
        <Card key={s.id as string} className="glass-card">
          <CardHeader className="pb-2">
            <div className="flex items-start justify-between">
              <div>
                <CardTitle className="text-base flex items-center gap-2">
                  <Heart className="h-4 w-4 text-primary" />
                  {s.session_type as string} Session
                  {s.therapist_name && <span className="text-muted-foreground font-normal">with {s.therapist_name as string}</span>}
                </CardTitle>
                <CardDescription>
                  {format(parseISO(s.session_date as string), "EEEE, MMMM d, yyyy")}
                  {s.duration_minutes && ` · ${s.duration_minutes}min`}
                </CardDescription>
              </div>
              <Button size="icon" variant="ghost" onClick={() => handleDelete(s.id as string)} disabled={isPending}>
                <Trash2 className="h-4 w-4 text-destructive" />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            {(s.mood_before !== null || s.mood_after !== null) && (
              <div className="flex items-center gap-2 text-sm">
                <span>Mood: {MOOD_LABELS[(s.mood_before as number) ?? 0]}</span>
                <ArrowRight className="h-3 w-3" />
                <span>{MOOD_LABELS[(s.mood_after as number) ?? 0]}</span>
              </div>
            )}
            {s.notes && <p className="text-sm">{s.notes as string}</p>}
            {s.homework_assigned && (
              <p className="text-sm text-muted-foreground">Homework: {s.homework_assigned as string}</p>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
