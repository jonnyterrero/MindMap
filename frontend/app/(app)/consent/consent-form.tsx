"use client";

import { useState, useTransition } from "react";
import { grantConsent } from "./actions";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Loader2, Shield } from "lucide-react";

const CONSENT_ITEMS = [
  {
    id: "terms_of_service",
    label: "Terms of Service",
    description: "I agree to the MindMap Terms of Service and understand this is a wellness tracking tool.",
    required: true,
  },
  {
    id: "data_collection",
    label: "Data Collection",
    description: "I understand that my self-reported mental health data is stored securely and used only to provide personalized insights.",
    required: true,
  },
  {
    id: "data_sharing_notice",
    label: "Data Sharing",
    description: "I understand that my data is never shared without my explicit consent. I control who sees my data and can revoke access at any time.",
    required: true,
  },
  {
    id: "analytics_opt_in",
    label: "Anonymous Analytics (optional)",
    description: "I allow anonymized, aggregated usage data to improve the app. No personal health data is included.",
    required: false,
  },
];

export function ConsentForm() {
  const [isPending, startTransition] = useTransition();
  const [checked, setChecked] = useState<Record<string, boolean>>({});

  const requiredItems = CONSENT_ITEMS.filter((c) => c.required);
  const allRequiredChecked = requiredItems.every((c) => checked[c.id]);

  function handleSubmit() {
    const consented = Object.entries(checked)
      .filter(([, v]) => v)
      .map(([k]) => k);
    startTransition(async () => {
      await grantConsent(consented);
    });
  }

  return (
    <Card className="glass-card">
      <CardHeader className="text-center">
        <div className="flex justify-center mb-3">
          <div className="p-3 rounded-full bg-primary/10">
            <Brain className="h-8 w-8 text-primary" />
          </div>
        </div>
        <CardTitle className="text-xl">Welcome to MindMap</CardTitle>
        <CardDescription className="max-w-sm mx-auto">
          Before you begin tracking, please review and accept the following.
          Your data privacy is our top priority.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {CONSENT_ITEMS.map((item) => (
          <div key={item.id} className="flex items-start gap-3 p-3 rounded-lg hover:bg-muted/50 transition-colors">
            <Checkbox
              id={item.id}
              checked={checked[item.id] ?? false}
              onCheckedChange={(v) =>
                setChecked((prev) => ({ ...prev, [item.id]: !!v }))
              }
              disabled={isPending}
              className="mt-0.5"
            />
            <div className="space-y-1">
              <Label htmlFor={item.id} className="cursor-pointer font-medium flex items-center gap-1">
                {item.label}
                {item.required && <span className="text-destructive">*</span>}
              </Label>
              <p className="text-xs text-muted-foreground">{item.description}</p>
            </div>
          </div>
        ))}

        <div className="flex items-center gap-2 text-xs text-muted-foreground p-2">
          <Shield className="h-4 w-4 shrink-0" />
          <span>
            You can export or delete your data at any time from Settings.
          </span>
        </div>

        <Button
          onClick={handleSubmit}
          disabled={isPending || !allRequiredChecked}
          className="w-full"
          size="lg"
        >
          {isPending ? <Loader2 className="animate-spin" /> : "Get Started"}
        </Button>
      </CardContent>
    </Card>
  );
}
