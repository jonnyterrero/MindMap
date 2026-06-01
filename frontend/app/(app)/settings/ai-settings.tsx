"use client";

import { useState, useTransition } from "react";
import { updateAiReflectionSetting } from "./actions";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { Sparkles } from "lucide-react";

export function AiSettings({ enabled }: { enabled: boolean }) {
  const [on, setOn] = useState(enabled);
  const [isPending, startTransition] = useTransition();
  const [err, setErr] = useState<string | null>(null);

  function toggle(next: boolean) {
    setOn(next);
    setErr(null);
    startTransition(async () => {
      const r = await updateAiReflectionSetting(next);
      if (r?.error) {
        setErr(r.error);
        setOn(!next); // revert on failure
      }
    });
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <Sparkles className="h-5 w-5 text-primary" /> AI journal reflection
        </CardTitle>
        <CardDescription>
          Optional. Adds a &ldquo;Reflect with AI&rdquo; button to each journal entry that
          offers a gentle summary, a reflection question, and emotional-theme tags.
          Your entry text is sent to Anthropic only when you tap that button.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <Label htmlFor="ai-reflection-toggle">Enable AI reflection</Label>
          <Switch
            id="ai-reflection-toggle"
            checked={on}
            onCheckedChange={toggle}
            disabled={isPending}
          />
        </div>
        {err && <p className="text-sm text-destructive">{err}</p>}
        <MedicalDisclaimer variant="inline" />
      </CardContent>
    </Card>
  );
}
