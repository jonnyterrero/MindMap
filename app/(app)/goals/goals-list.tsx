"use client";

import { useState, useTransition } from "react";
import { createGoal, toggleGoalComplete, deleteGoal, type GoalPayload } from "./actions";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Card, CardContent, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Plus, Trash2, Loader2, Target } from "lucide-react";

type Goal = Record<string, unknown>;

const CATEGORIES = ["Sleep", "Mood", "Exercise", "Productivity", "Medication", "Therapy", "Other"];

export function GoalsList({ goals: initialGoals }: { goals: Goal[] }) {
  const [isPending, startTransition] = useTransition();
  const [goals, setGoals] = useState(initialGoals);
  const [showNew, setShowNew] = useState(false);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [category, setCategory] = useState("Other");
  const [targetValue, setTargetValue] = useState("");
  const [unit, setUnit] = useState("");
  const [targetDate, setTargetDate] = useState("");

  function resetForm() {
    setTitle(""); setDescription(""); setCategory("Other");
    setTargetValue(""); setUnit(""); setTargetDate("");
    setShowNew(false);
  }

  function handleCreate() {
    if (!title.trim()) return;
    const payload: GoalPayload = {
      title: title.trim(),
      description: description.trim() || null,
      category,
      target_value: targetValue ? Number(targetValue) : null,
      unit: unit.trim() || null,
      target_date: targetDate || null,
    };

    const optimistic: Goal = {
      id: `temp-${Date.now()}`,
      ...payload,
      current_value: 0,
      is_completed: false,
      is_active: true,
      created_at: new Date().toISOString(),
    };
    setGoals((prev) => [...prev, optimistic]);
    resetForm();

    startTransition(async () => {
      await createGoal(payload);
    });
  }

  function handleToggle(id: string, current: boolean) {
    setGoals((prev) =>
      prev.map((g) => (g.id === id ? { ...g, is_completed: !current } : g))
    );
    startTransition(async () => {
      await toggleGoalComplete(id, !current);
    });
  }

  function handleDelete(id: string) {
    setGoals((prev) => prev.filter((g) => g.id !== id));
    startTransition(async () => {
      await deleteGoal(id);
    });
  }

  const active = goals.filter((g) => !(g.is_completed as boolean));
  const completed = goals.filter((g) => g.is_completed as boolean);

  return (
    <div className="space-y-4">
      {!showNew ? (
        <Button onClick={() => setShowNew(true)}>
          <Plus className="h-4 w-4" /> New Goal
        </Button>
      ) : (
        <Card className="glass-card">
          <CardHeader><CardTitle className="text-base">New Goal</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Title</Label>
              <Input placeholder="e.g. Sleep 8 hours nightly" value={title} onChange={(e) => setTitle(e.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Description (optional)</Label>
              <Textarea placeholder="Why this goal matters..." value={description} onChange={(e) => setDescription(e.target.value)} rows={2} />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Category</Label>
                <Select value={category} onValueChange={setCategory}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {CATEGORIES.map((c) => (<SelectItem key={c} value={c}>{c}</SelectItem>))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Target Date (optional)</Label>
                <Input type="date" value={targetDate} onChange={(e) => setTargetDate(e.target.value)} />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Target Value (optional)</Label>
                <Input type="number" placeholder="e.g. 30" value={targetValue} onChange={(e) => setTargetValue(e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>Unit (optional)</Label>
                <Input placeholder="e.g. days, hours, sessions" value={unit} onChange={(e) => setUnit(e.target.value)} />
              </div>
            </div>
            <div className="flex gap-2">
              <Button onClick={handleCreate} disabled={isPending || !title.trim()}>
                {isPending ? <Loader2 className="animate-spin" /> : <Plus className="h-4 w-4" />} Create
              </Button>
              <Button variant="ghost" onClick={resetForm}>Cancel</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {active.length === 0 && completed.length === 0 && !showNew && (
        <p className="text-center text-muted-foreground py-8">No goals yet. Set your first one above.</p>
      )}

      {active.length > 0 && (
        <div className="space-y-2">
          {active.map((goal) => {
            const target = goal.target_value as number | null;
            const current = goal.current_value as number | null;
            const pct = target && current ? Math.min((current / target) * 100, 100) : 0;
            return (
              <Card key={goal.id as string} className="glass-card">
                <CardContent className="flex items-start gap-3 py-4 px-4">
                  <Checkbox
                    checked={false}
                    onCheckedChange={() => handleToggle(goal.id as string, false)}
                    disabled={isPending}
                    className="mt-1"
                  />
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-2">
                      <Target className="h-4 w-4 text-primary shrink-0" />
                      <span className="font-medium">{goal.title as string}</span>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary">
                        {goal.category as string}
                      </span>
                    </div>
                    {goal.description && (
                      <p className="text-sm text-muted-foreground">{goal.description as string}</p>
                    )}
                    {target && (
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>{current ?? 0} / {target} {goal.unit as string ?? ""}</span>
                          <span>{Math.round(pct)}%</span>
                        </div>
                        <Progress value={pct} className="h-1.5" />
                      </div>
                    )}
                  </div>
                  <Button size="icon" variant="ghost" onClick={() => handleDelete(goal.id as string)} disabled={isPending}>
                    <Trash2 className="h-4 w-4 text-destructive" />
                  </Button>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {completed.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-sm font-medium text-muted-foreground">Completed</h2>
          {completed.map((goal) => (
            <Card key={goal.id as string} className="glass-card opacity-60">
              <CardContent className="flex items-center gap-3 py-3 px-4">
                <Checkbox
                  checked={true}
                  onCheckedChange={() => handleToggle(goal.id as string, true)}
                  disabled={isPending}
                />
                <span className="flex-1 line-through text-muted-foreground">{goal.title as string}</span>
                <Button size="icon" variant="ghost" onClick={() => handleDelete(goal.id as string)} disabled={isPending}>
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
