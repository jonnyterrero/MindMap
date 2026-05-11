"use client";

import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";

type Sensation = {
  id: string;
  body_part: string;
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
  { id: "Full Body", label: "Full body", cx: 50, cy: 50, r: 0 }, // never plotted
] as const;

type RegionId = (typeof REGIONS)[number]["id"];

function intensityColor(avg: number): string {
  // 0..10 → blue (cool) to red (hot)
  if (avg === 0) return "rgba(120, 120, 120, 0.15)";
  if (avg < 3) return "rgba(96, 165, 250, 0.55)"; // blue-400
  if (avg < 5) return "rgba(250, 204, 21, 0.65)"; // yellow-400
  if (avg < 7) return "rgba(251, 146, 60, 0.7)"; // orange-400
  return "rgba(239, 68, 68, 0.8)"; // red-500
}

export function BodyMapView({ sensations }: { sensations: Sensation[] }) {
  const [selected, setSelected] = useState<RegionId | null>(null);

  // Aggregate counts + average intensity per region
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

  if (sensations.length === 0) {
    return (
      <p className="py-12 text-center text-sm text-muted-foreground">
        No body sensations logged in the last 30 days. Log one from{" "}
        <a href="/today" className="underline">
          Today
        </a>{" "}
        to see it here.
      </p>
    );
  }

  const selectedRows = selected
    ? sensations.filter((s) => s.body_part === selected)
    : [];

  return (
    <div className="grid gap-6 sm:grid-cols-[minmax(0,1fr)_240px]">
      <div className="relative mx-auto w-full max-w-[280px]">
        <svg
          viewBox="0 0 100 100"
          className="h-auto w-full"
          role="img"
          aria-label="Body map"
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
                  onClick={() => setSelected(region.id as RegionId)}
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

      <div className="space-y-3">
        <h3 className="text-sm font-medium">
          {selected ? selected : "Tap a region"}
        </h3>
        {!selected && (
          <p className="text-xs text-muted-foreground">
            Colour intensity reflects average reported severity (0–10).
            Tap a region for recent entries.
          </p>
        )}
        {selected && selectedRows.length === 0 && (
          <p className="text-xs text-muted-foreground">
            No entries for {selected} in the last 30 days.
          </p>
        )}
        {selected && selectedRows.length > 0 && (
          <ul className="space-y-2 text-sm">
            {selectedRows.slice(0, 8).map((s) => (
              <li
                key={s.id}
                className="rounded-md border border-border/60 p-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">
                    {s.intensity}/10
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {new Date(s.created_at).toLocaleDateString()}
                  </span>
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
    </div>
  );
}
