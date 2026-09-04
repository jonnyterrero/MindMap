"use client";

import { useState, useTransition, type ReactNode } from "react";
import Link from "next/link";
import { grantConsent } from "./actions";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Loader2, Shield } from "lucide-react";

// Small helper so the description bodies can include real anchor tags without
// interpolating raw href strings. `target="_blank"` keeps the consent form
// state intact if the user opens a doc for reference.
function DocLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <Link
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="font-medium underline underline-offset-2 hover:text-foreground"
    >
      {children}
    </Link>
  );
}

// IDs must match the CHECK constraint on consent_records.consent_type:
//   terms_of_service | privacy_policy | data_sharing |
//   analytics_collection | email_notifications | push_notifications
// The medical disclaimer doesn't have its own consent_type (adding one would
// need a migration, which the Phase 0/1 freeze avoids); it's acknowledged
// alongside Terms of Service, whose description now links to it explicitly.
const CONSENT_ITEMS: Array<{
  id: string;
  label: string;
  description: ReactNode;
  required: boolean;
}> = [
  {
    id: "terms_of_service",
    label: "Terms of Service and Medical Disclaimer",
    description: (
      <>
        I agree to the <DocLink href="/terms">Terms of Service</DocLink> and
        have read the <DocLink href="/medical-disclaimer">Medical Disclaimer</DocLink>.
        I understand MindMap is a self-tracking tool and does not diagnose,
        treat, or replace professional medical advice.
      </>
    ),
    required: true,
  },
  {
    id: "privacy_policy",
    label: "Privacy Policy and AI Disclosure",
    description: (
      <>
        I&apos;ve read the <DocLink href="/privacy">Privacy Policy</DocLink> and
        the <DocLink href="/ai-disclosure">AI Disclosure</DocLink>, and
        understand that my self-reported data is stored securely and used only
        to power my own insights.
      </>
    ),
    required: true,
  },
  {
    id: "data_sharing",
    label: "Data Sharing",
    description: (
      <>
        I understand that my data is never shared without my explicit consent.
        I control who sees my data and can revoke access at any time from{" "}
        <DocLink href="/data-deletion">Settings</DocLink>.
      </>
    ),
    required: true,
  },
  {
    id: "analytics_collection",
    label: "Anonymous Analytics (optional)",
    description: (
      <>
        I allow anonymized, aggregated usage data to improve the app. No
        personal health data is included.
      </>
    ),
    required: false,
  },
];

export function ConsentForm() {
  const [isPending, startTransition] = useTransition();
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);

  const requiredItems = CONSENT_ITEMS.filter((c) => c.required);
  const allRequiredChecked = requiredItems.every((c) => checked[c.id]);

  function handleSubmit() {
    setSubmitError(null);
    const consented = Object.entries(checked)
      .filter(([, v]) => v)
      .map(([k]) => k);
    startTransition(async () => {
      // grantConsent calls redirect() on success, which throws NEXT_REDIRECT
      // and never reaches this line. We only get a return value on failure.
      const result = await grantConsent(consented);
      if (result?.error) setSubmitError(result.error);
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
              <div className="text-xs text-muted-foreground leading-relaxed">{item.description}</div>
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

        {submitError && (
          <p className="text-sm text-destructive text-center">{submitError}</p>
        )}
      </CardContent>
    </Card>
  );
}
