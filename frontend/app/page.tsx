import Link from "next/link";
import {
  Brain,
  HeartPulse,
  LineChart,
  NotebookPen,
  MoonStar,
  Activity,
  Watch,
  ShieldCheck,
  Lock,
  Download,
  Trash2,
  Sparkles,
  Waypoints,
  GaugeCircle,
} from "lucide-react";
import { createClient } from "@/lib/supabase-server";
import { GlassButton, GlassPanel } from "@/components/glass";
import { LandingHero } from "@/components/landing/landing-hero";
import { PublicFooter } from "@/components/landing/public-footer";

const trustItems = [
  { icon: Lock, label: "Private self-tracking" },
  { icon: Sparkles, label: "Pattern discovery" },
  { icon: ShieldCheck, label: "Not medical advice" },
];

const features = [
  {
    icon: HeartPulse,
    title: "Track what matters",
    body: "Log mood, anxiety, focus, sleep, and medication in a few calm taps each day.",
  },
  {
    icon: Activity,
    title: "Symptoms & migraines",
    body: "Note headaches, body sensations, and possible triggers as they happen.",
  },
  {
    icon: LineChart,
    title: "Trends & signals",
    body: "Your entries surface as trends over weeks and months, easier to notice and reflect on.",
  },
  {
    icon: NotebookPen,
    title: "Journaling & therapy",
    body: "Pair free-form notes and therapy reflections with structured check-ins.",
  },
  {
    icon: MoonStar,
    title: "Sleep & routines",
    body: "Line up sleep and daily routines next to how you feel to explore relationships.",
  },
  {
    icon: Watch,
    title: "Wearables & environment",
    body: "Bring in wearable data and environmental signals for a fuller picture over time.",
  },
];

const steps = [
  {
    n: "1",
    title: "Check in daily",
    body: "A quick, calm check-in captures mood, sleep, symptoms, and what's on your mind.",
  },
  {
    n: "2",
    title: "Build your history",
    body: "Entries quietly accumulate into a private, organized record that belongs to you.",
  },
  {
    n: "3",
    title: "Reflect on patterns",
    body: "Review trends and possible correlations to understand your patterns with more clarity.",
  },
];

const mlPoints = [
  {
    icon: Waypoints,
    title: "Correlations, not conclusions",
    body: "MindMap looks for statistical relationships in the data you log — like sleep next to mood — and shows them as possible patterns to explore.",
  },
  {
    icon: GaugeCircle,
    title: "Only your own data",
    body: "Insights are computed from the entries you choose to record. Nothing is inferred about you from outside sources.",
  },
  {
    icon: Sparkles,
    title: "Plain-language reflections",
    body: "Signals are described in clear, everyday language — never as a diagnosis, prediction of illness, or treatment recommendation.",
  },
];

const privacyPoints = [
  {
    icon: Lock,
    title: "Yours by default",
    body: "Your logs are private to your account. You decide what to track and what to leave out.",
  },
  {
    icon: Download,
    title: "Export anytime",
    body: "Take your data with you in a portable format whenever you want it.",
  },
  {
    icon: Trash2,
    title: "Delete on request",
    body: "Remove your data and account whenever you choose — no lock-in.",
  },
];

const DISCLAIMER_COPY =
  "MindMap is for self-tracking, journaling, wellness reflection, and personal pattern discovery only. It is not a medical device and does not diagnose, treat, cure, or prevent any disease or condition. Insights are based only on the data you choose to log and may be incomplete or inaccurate.";

export default async function LandingPage() {
  // The landing page is the first page every visitor sees — including
  // signed-in users. We only branch the CTAs based on auth state.
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  const signedIn = Boolean(user);

  return (
    <div data-app-theme="aurora" className="app-bg min-h-screen">
      {/* Public top nav */}
      <header className="sticky top-0 z-40 glass-dock">
        <div className="container mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
          <Link href="/" className="flex items-center gap-2 font-semibold">
            <Brain className="h-5 w-5 text-primary" aria-hidden="true" />
            <span className="font-[family-name:var(--font-space-grotesk)]">
              MindMap
            </span>
          </Link>
          <div className="flex items-center gap-2">
            {signedIn ? (
              <>
                <Link
                  href="/today"
                  className="rounded-full px-3 py-1.5 text-sm font-medium text-foreground hover:text-primary"
                >
                  Today&apos;s check-in
                </Link>
                <GlassButton asChild glow className="px-4 py-2">
                  <Link href="/home">Continue</Link>
                </GlassButton>
              </>
            ) : (
              <>
                <Link
                  href="/login"
                  className="rounded-full px-3 py-1.5 text-sm font-medium text-foreground hover:text-primary"
                >
                  Sign in
                </Link>
                <GlassButton asChild glow className="px-4 py-2">
                  <Link href="/signup">Create account</Link>
                </GlassButton>
              </>
            )}
          </div>
        </div>
      </header>

      <main>
        <LandingHero signedIn={signedIn} />

        {/* Trust strip */}
        <section className="container mx-auto max-w-5xl px-4 pb-4">
          <GlassPanel className="flex flex-col items-center justify-center gap-3 p-4 sm:flex-row sm:gap-8">
            {trustItems.map(({ icon: Icon, label }) => (
              <div
                key={label}
                className="flex items-center gap-2 text-sm font-medium text-muted-foreground"
              >
                <Icon className="h-4 w-4 text-primary" aria-hidden="true" />
                {label}
              </div>
            ))}
          </GlassPanel>
        </section>

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
        <section
          id="how"
          className="container mx-auto max-w-5xl scroll-mt-20 px-4 py-12"
        >
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

        {/* ML / correlation explanation */}
        <section className="container mx-auto max-w-5xl px-4 py-12">
          <div className="mx-auto max-w-2xl text-center">
            <span className="inline-flex items-center gap-2 rounded-full border border-white/40 bg-white/55 px-3 py-1 text-xs font-medium text-muted-foreground backdrop-blur-xl">
              <Sparkles className="h-3.5 w-3.5 text-primary" aria-hidden="true" />
              How pattern discovery works
            </span>
            <h2 className="mt-4 text-balance text-2xl font-bold tracking-tight sm:text-3xl font-[family-name:var(--font-space-grotesk)]">
              Signals surfaced from your own data
            </h2>
            <p className="mt-3 text-pretty text-muted-foreground">
              MindMap highlights possible relationships in what you log. These
              are observations to reflect on — not medical findings.
            </p>
          </div>

          <div className="mt-10 grid gap-4 md:grid-cols-3">
            {mlPoints.map(({ icon: Icon, title, body }) => (
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

        {/* Privacy / data control */}
        <section className="container mx-auto max-w-5xl px-4 py-12">
          <GlassPanel className="grid items-center gap-8 p-8 sm:p-10 md:grid-cols-2">
            <div className="space-y-4">
              <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary/15 text-primary">
                <ShieldCheck className="h-6 w-6" aria-hidden="true" />
              </span>
              <h2 className="text-balance text-2xl font-bold tracking-tight sm:text-3xl font-[family-name:var(--font-space-grotesk)]">
                Your data stays yours
              </h2>
              <p className="text-pretty text-muted-foreground">
                MindMap is built privacy-first. You control what you track,
                you can export your history at any time, and you can delete
                everything whenever you decide.
              </p>
              <GlassButton asChild>
                <Link href="/data-deletion">Learn about data control</Link>
              </GlassButton>
            </div>

            <ul className="grid gap-3">
              {privacyPoints.map(({ icon: Icon, title, body }) => (
                <li
                  key={title}
                  className="flex gap-3 rounded-xl border border-white/30 bg-white/40 p-4 backdrop-blur-xl"
                >
                  <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
                    <Icon className="h-4 w-4" aria-hidden="true" />
                  </span>
                  <div>
                    <h3 className="text-sm font-semibold">{title}</h3>
                    <p className="text-sm leading-relaxed text-muted-foreground">
                      {body}
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          </GlassPanel>
        </section>

        {/* Medical disclaimer card */}
        <section className="container mx-auto max-w-3xl px-4 py-4">
          <aside
            role="note"
            aria-label="Medical disclaimer"
            className="rounded-2xl border border-amber-300/50 bg-amber-50/60 p-5 text-sm text-amber-900 dark:border-amber-500/30 dark:bg-amber-950/30 dark:text-amber-100"
          >
            <div className="flex gap-3">
              <ShieldCheck
                className="mt-0.5 h-5 w-5 shrink-0"
                aria-hidden="true"
              />
              <div className="space-y-1">
                <p className="font-semibold">Medical disclaimer</p>
                <p className="leading-relaxed">{DISCLAIMER_COPY}</p>
              </div>
            </div>
          </aside>
        </section>

        {/* Final CTA */}
        <section className="container mx-auto max-w-5xl px-4 py-12">
          <GlassPanel className="flex flex-col items-center gap-5 p-8 text-center sm:p-12">
            <h2 className="text-balance text-2xl font-bold tracking-tight sm:text-3xl font-[family-name:var(--font-space-grotesk)]">
              {signedIn
                ? "Ready to continue?"
                : "Start understanding your patterns"}
            </h2>
            <p className="max-w-xl text-pretty text-muted-foreground">
              {signedIn
                ? "Jump back into your private record and log today's check-in."
                : "Create a free account and begin your first check-in. Your data stays private and is always yours."}
            </p>
            <div className="flex flex-col gap-3 sm:flex-row">
              {signedIn ? (
                <>
                  <GlassButton asChild glow>
                    <Link href="/home">Continue where you left off</Link>
                  </GlassButton>
                  <GlassButton asChild>
                    <Link href="/today">Go to today&apos;s check-in</Link>
                  </GlassButton>
                </>
              ) : (
                <>
                  <GlassButton asChild glow>
                    <Link href="/signup">Create account</Link>
                  </GlassButton>
                  <GlassButton asChild>
                    <Link href="/login">Sign in</Link>
                  </GlassButton>
                </>
              )}
            </div>
          </GlassPanel>
        </section>
      </main>

      <PublicFooter />
    </div>
  );
}
