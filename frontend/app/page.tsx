import Link from "next/link";
import { redirect } from "next/navigation";
import {
  Brain, HeartPulse, LineChart, NotebookPen, MoonStar,
  Pill, MessagesSquare,
} from "lucide-react";
import { createClient } from "@/lib/supabase-server";
import { GlassButton, GlassPanel } from "@/components/glass";
import { LandingHero } from "@/components/landing/landing-hero";
import { PublicFooter } from "@/components/landing/public-footer";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";

const features = [
  {
    icon: HeartPulse,
    title: "Track what matters",
    body: "Log mood, anxiety, focus, sleep, medication, routines, and body sensations in a few taps.",
  },
  {
    icon: LineChart,
    title: "See trends and signals",
    body: "Patterns surface over weeks and months, so changes are easier to notice and reflect on.",
  },
  {
    icon: NotebookPen,
    title: "Journal with context",
    body: "Pair free-form notes with structured check-ins to organize how you're really doing.",
  },
  {
    icon: MoonStar,
    title: "Connect sleep & routines",
    body: "Line up sleep, habits, and daily routines next to how you feel to spot relationships.",
  },
  {
    icon: Pill,
    title: "Keep medication in view",
    body: "Track medications and therapy alongside your check-ins for a fuller picture over time.",
  },
  {
    icon: MessagesSquare,
    title: "Support better conversations",
    body: "Bring clear, organized trends to appointments to support conversations with professionals.",
  },
];

const steps = [
  { n: "1", title: "Check in daily", body: "A quick, calm check-in captures mood, sleep, and what's on your mind." },
  { n: "2", title: "Build your history", body: "Entries quietly accumulate into a private, organized record over time." },
  { n: "3", title: "Reflect on patterns", body: "Review trends and signals to understand your patterns with more clarity." },
];

export default async function LandingPage() {
  // Signed-in users skip the marketing page and go straight to the app.
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (user) redirect("/home");

  return (
    <div data-app-theme="aurora" className="app-bg min-h-screen">
      {/* Public top nav */}
      <header className="sticky top-0 z-40 glass-dock">
        <div className="container mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
          <Link href="/" className="flex items-center gap-2 font-semibold">
            <Brain className="h-5 w-5 text-primary" aria-hidden="true" />
            <span className="font-[family-name:var(--font-space-grotesk)]">MindMap</span>
          </Link>
          <div className="flex items-center gap-2">
            <Link
              href="/login"
              className="rounded-full px-3 py-1.5 text-sm font-medium text-foreground hover:text-primary"
            >
              Sign in
            </Link>
            <GlassButton asChild glow className="px-4 py-2">
              <Link href="/signup">Get Started</Link>
            </GlassButton>
          </div>
        </div>
      </header>

      <main>
        <LandingHero />

        {/* Features */}
        <section className="container mx-auto max-w-5xl px-4 py-12">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-balance text-2xl font-bold tracking-tight sm:text-3xl font-[family-name:var(--font-space-grotesk)]">
              Everything in one calm, private place
            </h2>
            <p className="mt-3 text-pretty text-muted-foreground">
              MindMap organizes the signals that shape how you feel — without
              the clutter, and without claiming to diagnose or treat.
            </p>
          </div>

          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {features.map(({ icon: Icon, title, body }) => (
              <GlassPanel key={title} className="p-5">
                <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/15 text-primary">
                  <Icon className="h-5 w-5" aria-hidden="true" />
                </span>
                <h3 className="mt-3 font-semibold">{title}</h3>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  {body}
                </p>
              </GlassPanel>
            ))}
          </div>
        </section>

        {/* How it works */}
        <section id="how" className="container mx-auto max-w-5xl scroll-mt-20 px-4 py-12">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-balance text-2xl font-bold tracking-tight sm:text-3xl font-[family-name:var(--font-space-grotesk)]">
              How it works
            </h2>
            <p className="mt-3 text-pretty text-muted-foreground">
              A gentle daily rhythm that turns small check-ins into meaningful
              long-term context.
            </p>
          </div>

          <div className="mt-10 grid gap-4 md:grid-cols-3">
            {steps.map(({ n, title, body }) => (
              <GlassPanel key={n} className="p-6">
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                  {n}
                </span>
                <h3 className="mt-3 font-semibold">{title}</h3>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  {body}
                </p>
              </GlassPanel>
            ))}
          </div>
        </section>

        {/* CTA */}
        <section className="container mx-auto max-w-5xl px-4 py-12">
          <GlassPanel className="flex flex-col items-center gap-5 p-8 text-center sm:p-12">
            <h2 className="text-balance text-2xl font-bold tracking-tight sm:text-3xl font-[family-name:var(--font-space-grotesk)]">
              Start understanding your patterns
            </h2>
            <p className="max-w-xl text-pretty text-muted-foreground">
              Create a free account and begin your first check-in. Your data
              stays private and is always yours.
            </p>
            <div className="flex flex-col gap-3 sm:flex-row">
              <GlassButton asChild glow>
                <Link href="/signup">Get Started</Link>
              </GlassButton>
              <GlassButton asChild>
                <Link href="/login">Sign in</Link>
              </GlassButton>
            </div>
          </GlassPanel>
        </section>

        {/* Safety / disclaimer */}
        <section className="container mx-auto max-w-3xl px-4 pb-16">
          <MedicalDisclaimer variant="full" />
        </section>
      </main>

      <PublicFooter />
    </div>
  );
}
