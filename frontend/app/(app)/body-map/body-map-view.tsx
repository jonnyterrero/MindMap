"use client";

import { useMemo, useState, useTransition } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, Plus, Trash2 } from "lucide-react";
import { logBodySensation, removeBodySensation } from "./actions";

type Sensation = {
  id: string;
  body_part: string;
  sensation: string | null;
  intensity: number;
  notes: string | null;
  created_at: string;
  entry_id: string;
};

// Regions correspond to body_part strings written by the today page.
// Order matters: rendered top-to-bottom on the body silhouette.
const REGIONS = [
  { id: "Head", label: "Head", cx: 50, cy: 9, r: 7 },
  { id: "Jaw", label: "Jaw", cx: 50, cy: 17, r: 3 },
  { id: "Eyes", label: "Eyes", cx: 50, cy: 11, r: 2.5 },
  { id: "Neck", label: "Neck", cx: 50, cy: 21, r: 3 },
  { id: "Shoulders", label: "Shoulders", cx: 50, cy: 27, r: 9 },
  { id: "Chest", label: "Chest", cx: 50, cy: 36, r: 8 },
  { id: "Upper Back", label: "Upper back", cx: 50, cy: 36, r: 8 },
  { id: "Arms", label: "Arms", cx: 30, cy: 42, r: 5 },
  { id: "Hands", label: "Hands", cx: 22, cy: 55, r: 4 },
  { id: "Stomach", label: "Stomach", cx: 50, cy: 48, r: 7 },
  { id: "Lower Back", label: "Lower back", cx: 50, cy: 52, r: 7 },
  { id: "Legs", label: "Legs", cx: 50, cy: 70, r: 9 },
  { id: "Feet", label: "Feet", cx: 50, cy: 90, r: 5 },
  { id: "Full Body", label: "Full body", cx: 50, cy: 50, r: 0 }, // selectable via chip, never plotted
] as const;

type RegionId = (typeof REGIONS)[number]["id"];

// Sensation qualities a user can attach to a region. Must match the DB CHECK
// constraint on mindmap_body_sensations.sensation (migration 007) and the
// today-page form — any other value is rejected on insert.
const SENSATIONS = [
  "Pain",
  "Tension",
  "Numbness",
  "Tingling",
  "Burning",
  "Pressure",
  "Throbbing",
  "Aching",
  "Stiffness",
  "Heaviness",
  "Lightness",
  "Warmth",
  "Coldness",
  "Nausea",
] as const;

function intensityColor(avg: number): string {
  // 0..10 → blue (cool) to red (hot)
  if (avg === 0) return "rgba(120, 120, 120, 0.15)";
  if (avg < 3) return "rgba(96, 165, 250, 0.55)"; // blue-400
  if (avg < 5) return "rgba(250, 204, 21, 0.65)"; // yellow-400
  if (avg < 7) return "rgba(251, 146, 60, 0.7)"; // orange-400
  return "rgba(239, 68, 68, 0.8)"; // red-500
}

export function BodyMapView({ sensations: initial }: { sensations: Sensation[] }) {
  const [sensations, setSensations] = useState<Sensation[]>(initial);
  const [selected, setSelected] = useState<RegionId | null>(null);
  const [sensation, setSensation] = useState<string>(SENSATIONS[0]);
  const [intensity, setIntensity] = useState<number>(5);
  const [notes, setNotes] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  // Aggregate counts + average intensity per region (drives the heatmap).
  const stats = useMemo(() => {
    const byRegion = new Map<string, { count: number; sum: number }>();
    for (const s of sensations) {
      const r = byRegion.get(s.body_part) ?? { count: 0, sum: 0 };
      r.count += 1;
      r.sum += s.intensity;
      byRegion.set(s.body_part, r);
    }
    return byRegion;
  }, [sensations]);

  const selectedRows = selected
    ? sensations
        .filter((s) => s.body_part === selected)
        .sort((a, b) => b.created_at.localeCompare(a.created_at))
    : [];

  function pick(region: RegionId) {
    setSelected(region);
    setError(null);
  }

  function handleLog() {
    if (!selected) return;
    setError(null);

    const optimistic: Sensation = {
      id: `temp-${Date.now()}`,
      body_part: selected,
      sensation,
      intensity,
      notes: notes.trim() || null,
      created_at: new Date().toISOString(),
      entry_id: `temp-${Date.now()}`,
    };
    setSensations((prev) => [optimistic, ...prev]);
    const noteValue = notes.trim() || null;
    setNotes("");

    startTransition(async () => {
      const res = await logBodySensation({
        bodyPart: selected,
        sensation,
        intensity,
        notes: noteValue,
      });
      if ("error" in res) {
        setSensations((prev) => prev.filter((s) => s.id !== optimistic.id));
        setError(res.error);
      }
    });
  }

  function handleRemove(id: string) {
    const prev = sensations;
    setSensations((cur) => cur.filter((s) => s.id !== id));
    startTransition(async () => {
      const res = await removeBodySensation(id);
      if (res && "error" in res) {
        setSensations(prev); // revert on failure
        setError(res.error);
      }
    });
  }

  return (
    <div className="grid gap-6 sm:grid-cols-[minmax(0,1fr)_280px]">
      <div className="space-y-4">
        <div className="relative mx-auto w-full max-w-[280px]">
          <svg
            viewBox="0 0 100 100"
            className="h-auto w-full"
            role="img"
            aria-label="Body map — tap a region to log a sensation"
          >
            {/* Silhouette */}
            <path
              d="M50 4 a8 8 0 1 1 -0.1 0 z M42 18 h16 v6 h-16 z M30 25 h40 v18 l-8 5 v25 l5 35 h-10 l-3 -32 h-8 l-3 32 h-10 l5 -35 v-25 l-8 -5 z"
              fill="rgba(0,0,0,0.06)"
              stroke="rgba(0,0,0,0.25)"
              strokeWidth="0.4"
              className="dark:fill-white/5 dark:stroke-white/20"
            />
            {/* Region markers */}
            {REGIONS.filter((r) => r.r > 0).map((region) => {
              const s = stats.get(region.id);
              const avg = s ? s.sum / s.count : 0;
              const active = selected === region.id;
              return (
                <g key={region.id}>
                  <circle
                    cx={region.cx}
                    cy={region.cy}
                    r={region.r}
                    fill={s ? intensityColor(avg) : "transparent"}
                    stroke={active ? "currentColor" : "rgba(0,0,0,0.2)"}
                    strokeWidth={active ? 0.8 : 0.3}
                    className={cn(
                      "cursor-pointer transition-opacity",
                      s ? "opacity-100" : "opacity-30 hover:opacity-60",
                    )}
                    onClick={() => pick(region.id as RegionId)}
                  >
                    <title>
                      {region.label}
                      {s ? ` · ${s.count}× · avg ${avg.toFixed(1)}/10` : ""}
                    </title>
                  </circle>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Region chips (covers Full body + a tap-free fallback) */}
        <div className="flex flex-wrap justify-center gap-1.5">
          {REGIONS.map((region) => (
            <button
              key={region.id}
              type="button"
              onClick={() => pick(region.id as RegionId)}
              className={cn(
                "rounded-full border px-2.5 py-1 text-xs transition-colors",
                selected === region.id
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border/60 text-muted-foreground hover:border-border",
              )}
            >
              {region.label}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-4">
        {!selected ? (
          <div className="space-y-2">
            <h3 className="text-sm font-medium">Tap a region to log</h3>
            <p className="text-xs text-muted-foreground">
              Colour intensity reflects average reported severity (0–10) over the
              last 30 days. Tap a body area or chip to log a new sensation.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium">{selected}</h3>
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="text-xs text-muted-foreground underline"
              >
                Clear
              </button>
            </div>

            {/* Logging form */}
            <div className="space-y-3 rounded-lg border border-border/60 p-3">
              <div className="space-y-1.5">
                <Label className="text-xs">Sensation</Label>
                <div className="flex flex-wrap gap-1.5">
                  {SENSATIONS.map((sn) => (
                    <button
                      key={sn}
                      type="button"
                      onClick={() => setSensation(sn)}
                      className={cn(
                        "rounded-full border px-2.5 py-1 text-xs transition-colors",
                        sensation === sn
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border/60 text-muted-foreground hover:border-border",
                      )}
                    >
                      {sn}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Intensity</Label>
                  <span className="text-xs font-medium tabular-nums">
                    {intensity}/10
                  </span>
                </div>
                <Slider
                  min={0}
                  max={10}
                  step={1}
                  value={[intensity]}
                  onValueChange={(v: number[]) => setIntensity(v[0] ?? 0)}
                />
              </div>

              <div className="space-y-1.5">
                <Label className="text-xs">Note (optional)</Label>
                <Textarea
                  rows={2}
                  placeholder="e.g. worse after screen time"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                />
              </div>

              {error && <p className="text-xs text-destructive">{error}</p>}

              <Button
                size="sm"
                onClick={handleLog}
                disabled={isPending}
                className="w-full"
              >
                {isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Plus className="h-4 w-4" />
                )}
                Log {sensation.toLowerCase()} in {selected.toLowerCase()}
              </Button>
            </div>

            {/* Recent entries for this region */}
            {selectedRows.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                No entries for {selected} in the last 30 days.
              </p>
            ) : (
              <ul className="space-y-2 text-sm">
                {selectedRows.slice(0, 8).map((s) => (
                  <li
                    key={s.id}
                    className="rounded-md border border-border/60 p-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">
                        {s.sensation ? `${s.sensation} · ` : ""}
                        {s.intensity}/10
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">
                          {new Date(s.created_at).toLocaleDateString()}
                        </span>
                        {!s.id.startsWith("temp-") && (
                          <button
                            type="button"
                            onClick={() => handleRemove(s.id)}
                            disabled={isPending}
                            aria-label="Remove sensation"
                            className="text-muted-foreground hover:text-destructive disabled:opacity-50"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>
                    </div>
                    {s.notes && (
                      <p className="mt-1 text-xs text-muted-foreground">
                        {s.notes}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
