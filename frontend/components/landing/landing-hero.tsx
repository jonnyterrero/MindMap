import Link from "next/link";
import { GlassButton, GlassPanel } from "@/components/glass";
import {
  ArrowRight,
  Moon,
  Activity,
  BookOpen,
  TrendingUp,
} from "lucide-react";

const SIGNALS = [
  "Mood",
  "Sleep",
  "Focus",
  "Routines",
  "Medication",
  "Journaling",
  "Therapy",
  "Symptoms",
  "Wearables",
  "Environment",
];

/**
 * Landing hero. CTAs adapt to auth state:
 *  - signed out → Create account / See how it works
 *  - signed in  → Continue where you left off / Go to today's check-in
 */
export function LandingHero({ signedIn }: { signedIn: boolean }) {
  return (
    <section className="relative overflow-hidden">
      {/* Soft gradient accent behind the hero (never behind body text). */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-0 -top-24 mx-auto h-72 max-w-3xl rounded-full bg-gradient-to-br from-pink-400/30 via-violet-400/25 to-sky-400/25 blur-3xl"
      />

      <div className="container relative mx-auto grid max-w-5xl items-center gap-10 px-4 py-16 md:grid-cols-2 md:py-24">
        <div className="space-y-6">
          <span className="inline-flex items-center gap-2 rounded-full border border-white/40 bg-white/55 px-3 py-1 text-xs font-medium text-muted-foreground backdrop-blur-xl">
            Calm, private self-tracking
          </span>
          <h1 className="text-balance text-4xl font-bold leading-tight tracking-tight sm:text-5xl font-[family-name:var(--font-space-grotesk)]">
            {signedIn
              ? "Welcome back. Pick up where you left off."
              : "Notice your patterns in mood, sleep, and migraines."}
          </h1>
          <p className="text-pretty text-base leading-relaxed text-muted-foreground sm:text-lg">
            MindMap is a private space to log mood, sleep, focus, routines,
            medication, journaling, therapy, symptoms, wearable data, and
            environmental signals — so possible patterns become easier to
            notice over time.
          </p>

          {/* Signal chips */}
          <ul className="flex flex-wrap gap-2" aria-label="What you can track">
            {SIGNALS.map((s) => (
              <li
                key={s}
                className="rounded-full border border-white/40 bg-white/45 px-2.5 py-1 text-xs font-medium text-muted-foreground backdrop-blur-xl"
              >
                {s}
              </li>
            ))}
          </ul>

          <div className="flex flex-col gap-3 sm:flex-row">
            {signedIn ? (
              <>
                <GlassButton asChild glow>
                  <Link href="/home">
                    Continue where you left off
                    <ArrowRight className="h-4 w-4" aria-hidden="true" />
                  </Link>
                </GlassButton>
                <GlassButton asChild>
                  <Link href="/today">Go to today&apos;s check-in</Link>
                </GlassButton>
              </>
            ) : (
              <>
                <GlassButton asChild glow>
                  <Link href="/signup">
                    Create account
                    <ArrowRight className="h-4 w-4" aria-hidden="true" />
                  </Link>
                </GlassButton>
                <GlassButton asChild>
                  <Link href="#how">See how it works</Link>
                </GlassButton>
              </>
            )}
          </div>

          {!signedIn && (
            <p className="text-xs text-muted-foreground">
              Already have an account?{" "}
              <Link
                href="/login"
                className="font-medium text-primary hover:underline"
              >
                Sign in
              </Link>
            </p>
          )}
        </div>

        {/* Liquid-glass app preview mockup */}
        <div className="relative">
          <GlassPanel className="p-5">
            <div className="flex items-center justify-between">
              <p className="text-sm font-semibold">Today</p>
              <span className="rounded-full bg-primary/15 px-2.5 py-0.5 text-xs font-medium text-primary">
                Streak 12 days
              </span>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <PreviewStat icon={TrendingUp} label="Mood" value="Steady" />
              <PreviewStat icon={Moon} label="Sleep" value="7h 20m" />
              <PreviewStat icon={Activity} label="Migraine" value="None" />
              <PreviewStat icon={BookOpen} label="Journal" value="2 notes" />
            </div>

            <GlassPanel className="mt-4 p-4">
              <p className="text-xs font-medium text-muted-foreground">
                7-day mood trend
              </p>
              <div
                className="mt-3 flex h-16 items-end gap-1.5"
                aria-hidden="true"
              >
                {[40, 55, 48, 70, 62, 80, 72].map((h, i) => (
                  <span
                    key={i}
                    style={{ height: `${h}%` }}
                    className="flex-1 rounded-t-md bg-gradient-to-t from-primary/40 to-primary"
                  />
                ))}
              </div>
            </GlassPanel>
          </GlassPanel>
        </div>
      </div>
    </section>
  );
}

function PreviewStat({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Moon;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-white/30 bg-white/40 p-3 backdrop-blur-xl">
      <div className="flex items-center gap-1.5 text-muted-foreground">
        <Icon className="h-3.5 w-3.5" aria-hidden="true" />
        <span className="text-xs">{label}</span>
      </div>
      <p className="mt-1 text-sm font-semibold">{value}</p>
    </div>
  );
}
