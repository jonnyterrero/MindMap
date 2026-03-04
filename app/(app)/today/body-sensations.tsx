"use client";

import { useState, useTransition } from "react";
import { addBodySensation, removeBodySensation } from "./actions";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Plus, X, Loader2 } from "lucide-react";

type Sensation = {
  id: string;
  body_part: string;
  sensation: string;
  intensity: number;
};

const BODY_PARTS = [
  "Head", "Neck", "Shoulders", "Chest", "Upper Back",
  "Lower Back", "Stomach", "Arms", "Hands", "Legs",
  "Feet", "Jaw", "Eyes", "Full Body",
];

const SENSATIONS = [
  "Pain", "Tension", "Numbness", "Tingling", "Burning",
  "Pressure", "Throbbing", "Aching", "Stiffness", "Heaviness",
  "Lightness", "Warmth", "Coldness", "Nausea",
];

export function BodySensations({ sensations: initialSensations }: { sensations: Sensation[] }) {
  const [isPending, startTransition] = useTransition();
  const [sensations, setSensations] = useState(initialSensations);
  const [showAdd, setShowAdd] = useState(false);
  const [bodyPart, setBodyPart] = useState("Head");
  const [sensation, setSensation] = useState("Pain");
  const [intensity, setIntensity] = useState(5);

  function handleAdd() {
    const tempId = `temp-${Date.now()}`;
    const optimistic: Sensation = {
      id: tempId,
      body_part: bodyPart,
      sensation,
      intensity,
    };
    setSensations((prev) => [...prev, optimistic]);
    setShowAdd(false);
    setIntensity(5);

    startTransition(async () => {
      await addBodySensation(bodyPart, sensation, intensity);
    });
  }

  function handleRemove(id: string) {
    setSensations((prev) => prev.filter((s) => s.id !== id));
    startTransition(async () => {
      await removeBodySensation(id);
    });
  }

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="text-base">Body Sensations</CardTitle>
        <CardDescription>Track physical sensations you&apos;re experiencing</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {sensations.map((s) => (
          <div
            key={s.id}
            className="flex items-center justify-between text-sm p-2 rounded bg-muted/50"
          >
            <span>
              <strong>{s.body_part}</strong> — {s.sensation}{" "}
              <span className="text-muted-foreground">({s.intensity}/10)</span>
            </span>
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6"
              onClick={() => handleRemove(s.id)}
              disabled={isPending}
            >
              <X className="h-3 w-3" />
            </Button>
          </div>
        ))}

        {!showAdd ? (
          <Button variant="outline" size="sm" onClick={() => setShowAdd(true)}>
            <Plus className="h-3 w-3" /> Add Sensation
          </Button>
        ) : (
          <div className="space-y-3 p-3 border rounded-lg">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label className="text-xs">Body Part</Label>
                <Select value={bodyPart} onValueChange={setBodyPart}>
                  <SelectTrigger className="h-8 text-sm"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {BODY_PARTS.map((p) => <SelectItem key={p} value={p}>{p}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1">
                <Label className="text-xs">Sensation</Label>
                <Select value={sensation} onValueChange={setSensation}>
                  <SelectTrigger className="h-8 text-sm"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {SENSATIONS.map((s) => <SelectItem key={s} value={s}>{s}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between">
                <Label className="text-xs">Intensity</Label>
                <span className="text-xs text-muted-foreground">{intensity}/10</span>
              </div>
              <Slider min={1} max={10} step={1} value={[intensity]} onValueChange={([v]) => setIntensity(v)} />
            </div>
            <div className="flex gap-2">
              <Button size="sm" onClick={handleAdd} disabled={isPending}>
                {isPending ? <Loader2 className="animate-spin h-3 w-3" /> : <Plus className="h-3 w-3" />}
                Add
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setShowAdd(false)}>Cancel</Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
