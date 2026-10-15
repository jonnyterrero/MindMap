import type { Metadata } from "next";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { CalendarCheck, LineChart, ListChecks, Sparkles } from "lucide-react";

export const metadata: Metadata = {
  title: "You're all set · MindMap",
  description:
    "Your daily loop in MindMap: check in, build a 10-day baseline, and watch for personal patterns.",
};

// The daily loop, framed honestly: self-tracking first, "experimental pattern
// support" second — never clinical. Keep this copy aligned with
// docs/adr and DEVELOPMENT_PLAN 0.7.
const STEPS = [
  {
    icon: CalendarCheck,
    title: "Check in each day",
    body: "A daily check-in takes under 90 seconds. Consistency is what makes the patterns meaningful.",
  },
  {
    icon: ListChecks,
    title: "Build a 10-day baseline",
    body: "The first 10 days establish your personal baseline. Until then, MindMap holds off on trends rather than guessing from a day or two.",
  },
  {
    icon: LineChart,
    title: "Notice your patterns",
    body: "After the baseline, MindMap surfaces simple, experimental pattern support based only on what you log — observations to reflect on, never medical advice.",
  },
];

// Trust + support links gathered in one place as the user enters the product
// (DEVELOPMENT_PLAN 0.7: wire disclaimer + AI disclosure + privacy / deletion /
// support through the funnel). Every route already exists.
const LINKS = [
  { href: "/privacy", label: "Privacy" },
  { href: "/ai-disclosure", label: "AI & ML disclosure" },
  { href: "/medical-disclaimer", label: "Medical disclaimer" },
  { href: "/data-deletion", label: "Data & deletion" },
  { href: "/support", label: "Support" },
];

export default function NextStepsPage() {
  return (
    <div className="flex flex-1 flex-col justify-center gap-6">
      <Card>
        <CardHeader className="text-center">
          <div className="mb-3 flex justify-center">
            <div className="rounded-full bg-primary/10 p-3">
              <Sparkles className="h-8 w-8 text-primary" aria-hidden="true" />
            </div>
          </div>
          <CardTitle className="text-2xl">You&apos;re all set</CardTitle>
          <CardDescription className="mx-auto max-w-sm text-balance">
            Your check-in is ready. Here&apos;s how MindMap works day to day.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-5">
          <ol className="space-y-3">
            {STEPS.map(({ icon: Icon, title, body }, i) => (
              <li key={title} className="flex gap-3">
                <div
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary"
                  aria-hidden="true"
                >
                  <Icon className="h-4 w-4" />
                </div>
                <div className="space-y-0.5">
                  <p className="text-sm font-medium">
                    <span className="text-muted-foreground">{i + 1}.</span>{" "}
                    {title}
                  </p>
                  <p className="text-sm text-muted-foreground">{body}</p>
                </div>
              </li>
            ))}
          </ol>

          <div className="flex flex-col gap-2">
            <Button asChild size="lg" className="w-full">
              <Link href="/today">Start Day 1 Check-In</Link>
            </Button>
            <Button asChild variant="ghost" size="sm" className="w-full">
              <Link href="/home">Explore the app first</Link>
            </Button>
          </div>

          <MedicalDisclaimer variant="inline" />

          <nav
            aria-label="Privacy and support"
            className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 border-t pt-4 text-xs text-muted-foreground"
          >
            {LINKS.map((l) => (
              <Link
                key={l.href}
                href={l.href}
                className="underline-offset-4 hover:text-foreground hover:underline"
              >
                {l.label}
              </Link>
            ))}
          </nav>
        </CardContent>
      </Card>
    </div>
  );
}
