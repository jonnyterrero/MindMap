"use client";

import { useState, useTransition } from "react";
import {
  logMetric,
  disconnectSource,
  type WearableSourceRow,
  type RecentMetric,
} from "./wearable-actions";
import {
  METRIC_META,
  SOURCE_META,
  WEARABLE_METRICS,
  type WearableMetric,
  type WearableSourceType,
} from "@/lib/wearable";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Activity, Loader2, Plus, Trash2 } from "lucide-react";

export function WearableSettings({
  sources: initialSources,
  recentMetrics,
}: {
  sources: WearableSourceRow[];
  recentMetrics: RecentMetric[];
}) {
  const [sources, setSources] = useState(initialSources);
  const [metric, setMetric] = useState<WearableMetric>("hrv");
  const [value, setValue] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  function add() {
    setErr(null);
    setMsg(null);
    const v = Number(value);
    startTransition(async () => {
      const r = await logMetric(metric, v);
      if ("error" in r) setErr(r.error);
      else {
        setMsg(`Logged ${METRIC_META[metric].label}.`);
        setValue("");
      }
    });
  }

  function remove(id: string, type: string) {
    setSources((s) => s.filter((x) => x.source_type !== type));
    startTransition(async () => {
      await disconnectSource(id);
    });
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <Activity className="h-5 w-5 text-primary" /> Wearables &amp; health data
          <span className="ml-2 rounded-full bg-muted px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
            Manual entry
          </span>
        </CardTitle>
        <CardDescription>
          Log HRV and sleep score by hand — they feed your predictions. Automatic
          sync from Apple Health, Health Connect, and other devices is coming
          soon and isn&apos;t wired yet.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Manual entry */}
        <div className="space-y-2">
          <Label>Add a reading</Label>
          <div className="flex flex-wrap items-end gap-2">
            <Select value={metric} onValueChange={(v) => setMetric(v as WearableMetric)}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {WEARABLE_METRICS.map((m) => (
                  <SelectItem key={m} value={m}>
                    {METRIC_META[m].label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Input
              type="number"
              inputMode="decimal"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder={METRIC_META[metric].unit}
              className="w-28"
            />
            <Button onClick={add} disabled={isPending || !value.trim()}>
              {isPending ? <Loader2 className="animate-spin" /> : <Plus />} Add
            </Button>
          </div>
          {err && <p className="text-sm text-destructive">{err}</p>}
          {msg && <p className="text-sm text-green-600">{msg}</p>}
        </div>

        {/* Recent metrics */}
        {recentMetrics.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {recentMetrics.map((r) => (
              <span key={r.metric_type} className="rounded-full bg-muted px-2 py-0.5 text-xs">
                {METRIC_META[r.metric_type as WearableMetric]?.label ?? r.metric_type}: {r.value}
                {r.unit && r.unit !== "steps" ? ` ${r.unit}` : ""}
              </span>
            ))}
          </div>
        )}

        {/* Device sync -- deferred. The old "Connect Apple Health" / "Connect
            Health Connect" buttons wrote a stub row to mindmap_wearable_sources
            and did nothing else; no OAuth, no token exchange, no ingestion.
            Existing "connected" rows from before this pass stay visible so
            users can disconnect them; the connect buttons are gone. */}
        <div className="space-y-2 border-t pt-4">
          <div className="flex items-center gap-2">
            <Label>Device sync</Label>
            <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
              Coming soon
            </span>
          </div>
          {sources.length > 0 && (
            <div className="space-y-1.5">
              {sources.map((s) => (
                <div key={s.id} className="flex items-center justify-between rounded-lg border p-2.5">
                  <span className="text-sm">
                    {SOURCE_META[s.source_type as WearableSourceType]?.label ?? s.source_type}
                    <span className="ml-2 text-xs text-muted-foreground">(placeholder — no data flowed)</span>
                  </span>
                  <Button size="sm" variant="ghost" onClick={() => remove(s.id, s.source_type)} disabled={isPending}>
                    <Trash2 className="h-4 w-4 text-destructive" /> Remove
                  </Button>
                </div>
              ))}
            </div>
          )}
          <p className="text-xs text-muted-foreground">
            Native Apple Health and Health Connect sync will land in a future
            release. Until then, log the metrics you care about above.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
