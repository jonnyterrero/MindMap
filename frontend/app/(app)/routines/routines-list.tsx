"use client";

import { useState, useTransition } from "react";
import {
  createRoutine,
  updateRoutine,
  toggleRoutineActive,
  deleteRoutine,
} from "./actions";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Plus, Pencil, Trash2, Check, X, Loader2 } from "lucide-react";
import { availablePresets } from "@/lib/routine-presets";

type Routine = {
  id: string;
  name: string;
  is_active: boolean;
  created_at: string;
};

export function RoutinesList({ routines: initialRoutines }: { routines: Routine[] }) {
  const [newName, setNewName] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [isPending, startTransition] = useTransition();
  const [routines, setRoutines] = useState(initialRoutines);

  const presetGroups = availablePresets(routines.map((r) => r.name));

  /** Optimistically add a routine, then reconcile the temp row with the saved
   *  one (real id) or roll it back if the insert failed. Reconciling the id is
   *  what makes a just-added routine immediately toggle/rename/delete-able. */
  function addRoutine(name: string) {
    const trimmed = name.trim();
    if (!trimmed) return;
    const tempId = `temp-${Date.now()}`;
    const tempRoutine: Routine = {
      id: tempId,
      name: trimmed,
      is_active: true,
      created_at: new Date().toISOString(),
    };
    setRoutines((prev) => [...prev, tempRoutine]);

    startTransition(async () => {
      const result = await createRoutine(trimmed);
      setRoutines((prev) => {
        if ("routine" in result && result.routine) {
          return prev.map((r) => (r.id === tempId ? result.routine! : r));
        }
        // Insert failed — drop the optimistic row.
        return prev.filter((r) => r.id !== tempId);
      });
    });
  }

  function handleCreate() {
    if (!newName.trim()) return;
    addRoutine(newName);
    setNewName("");
  }

  function handleUpdate(id: string) {
    if (!editName.trim()) return;
    setRoutines((prev) =>
      prev.map((r) => (r.id === id ? { ...r, name: editName.trim() } : r))
    );
    setEditingId(null);

    startTransition(async () => {
      await updateRoutine(id, editName);
    });
  }

  function handleToggle(id: string, current: boolean) {
    setRoutines((prev) =>
      prev.map((r) => (r.id === id ? { ...r, is_active: !current } : r))
    );
    startTransition(async () => {
      await toggleRoutineActive(id, !current);
    });
  }

  function handleDelete(id: string) {
    setRoutines((prev) => prev.filter((r) => r.id !== id));
    startTransition(async () => {
      await deleteRoutine(id);
    });
  }

  return (
    <div className="space-y-4">
      {/* Add new */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-base">Add Routine</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {presetGroups.length > 0 && (
            <Select
              // key forces a remount so the trigger resets to placeholder after
              // each pick (a controlled-to-"" reset on Radix Select is awkward).
              key={routines.length}
              value=""
              onValueChange={(name) => addRoutine(name)}
              disabled={isPending}
            >
              <SelectTrigger className="w-full" aria-label="Add a suggested routine">
                <SelectValue placeholder="Choose a suggested routine…" />
              </SelectTrigger>
              <SelectContent>
                {presetGroups.map((group) => (
                  <SelectGroup key={group.category}>
                    <SelectLabel>{group.category}</SelectLabel>
                    {group.presets.map((preset) => (
                      <SelectItem key={preset.name} value={preset.name}>
                        {preset.name}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                ))}
              </SelectContent>
            </Select>
          )}

          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleCreate();
            }}
            className="flex gap-2"
          >
            <Input
              placeholder="e.g. Morning meditation, Exercise, Journaling..."
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              className="flex-1"
            />
            <Button type="submit" disabled={isPending || !newName.trim()}>
              {isPending ? (
                <Loader2 className="animate-spin" />
              ) : (
                <Plus className="h-4 w-4" />
              )}
              Add
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* List */}
      {routines.length === 0 ? (
        <p className="text-center text-muted-foreground py-8">
          No routines yet. Add your first one above.
        </p>
      ) : (
        <div className="space-y-2">
          {routines.map((routine) => (
            <Card key={routine.id} className="glass-card">
              <CardContent className="flex items-center gap-3 py-3 px-4">
                <Switch
                  checked={routine.is_active}
                  onCheckedChange={() =>
                    handleToggle(routine.id, routine.is_active)
                  }
                  disabled={isPending}
                />

                {editingId === routine.id ? (
                  <div className="flex items-center gap-2 flex-1">
                    <Input
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      className="flex-1"
                      autoFocus
                      onKeyDown={(e) => {
                        if (e.key === "Enter") handleUpdate(routine.id);
                        if (e.key === "Escape") setEditingId(null);
                      }}
                    />
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => handleUpdate(routine.id)}
                      disabled={isPending}
                    >
                      <Check className="h-4 w-4" />
                    </Button>
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => setEditingId(null)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ) : (
                  <>
                    <span
                      className={`flex-1 ${
                        !routine.is_active
                          ? "text-muted-foreground line-through"
                          : ""
                      }`}
                    >
                      {routine.name}
                    </span>
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => {
                        setEditingId(routine.id);
                        setEditName(routine.name);
                      }}
                    >
                      <Pencil className="h-4 w-4" />
                    </Button>
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => handleDelete(routine.id)}
                      disabled={isPending}
                    >
                      <Trash2 className="h-4 w-4 text-destructive" />
                    </Button>
                  </>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
