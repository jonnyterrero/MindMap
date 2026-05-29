"use client";

import { useState, useTransition } from "react";
import { updateWeatherSettings } from "./actions";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { CloudSun, Loader2 } from "lucide-react";

export function WeatherSettings({
  enabled,
  label,
}: {
  enabled: boolean;
  label: string | null;
}) {
  const [on, setOn] = useState(enabled);
  const [city, setCity] = useState(label ?? "");
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  function save() {
    setErr(null);
    setMsg(null);
    startTransition(async () => {
      const r = await updateWeatherSettings(on, city);
      if (r?.error) {
        setErr(r.error);
      } else {
        if (r?.label) setCity(r.label);
        setMsg(on ? `On${r?.label ? ` — ${r.label}` : ""}. Weather will appear in Insights.` : "Weather tracking turned off.");
      }
    });
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <CloudSun className="h-5 w-5 text-primary" /> Weather correlation
        </CardTitle>
        <CardDescription>
          Optional. Adds daily weather to your Insights so you can spot patterns like
          pressure changes vs. migraines. Uses your city only — no precise location stored.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <Label htmlFor="weather-toggle">Track weather</Label>
          <Switch id="weather-toggle" checked={on} onCheckedChange={setOn} disabled={isPending} />
        </div>

        {on && (
          <div className="space-y-2">
            <Label htmlFor="weather-city">City</Label>
            <Input
              id="weather-city"
              placeholder="e.g. Austin, TX"
              value={city}
              onChange={(e) => setCity(e.target.value)}
              disabled={isPending}
            />
          </div>
        )}

        <Button onClick={save} disabled={isPending} className="w-full sm:w-auto">
          {isPending ? <Loader2 className="animate-spin" /> : null}
          {isPending ? "Saving…" : "Save"}
        </Button>

        {err && <p className="text-sm text-destructive">{err}</p>}
        {msg && <p className="text-sm text-green-600">{msg}</p>}
      </CardContent>
    </Card>
  );
}
