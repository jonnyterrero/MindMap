/**
 * Correlation engine
 * ------------------
 * Pure, dependency-free Pearson correlations across a user's daily entries.
 * Intentionally conservative: requires a minimum sample size and a minimum
 * |r| before surfacing anything, and every result is phrased as a *possible
 * pattern*, never a cause or diagnosis.
 *
 * No external services, no API keys. Safe to run on the server per request.
 */

export interface MetricDef {
  key: string;
  label: string;
}

// Only numeric daily-entry fields. Higher-is-"worse" vs "better" is left out
// on purpose — we describe direction neutrally to avoid clinical framing.
export const CORRELATION_METRICS: MetricDef[] = [
  { key: "sleep_minutes", label: "Sleep duration" },
  { key: "sleep_quality", label: "Sleep quality" },
  { key: "mood_valence", label: "Mood" },
  { key: "anxiety", label: "Anxiety" },
  { key: "depression", label: "Depression" },
  { key: "focus", label: "Focus" },
  { key: "productivity", label: "Productivity" },
  { key: "migraine_intensity", label: "Migraine intensity" },
  // Body-map pain (max logged sensation intensity per day; merged in by
  // entry_date, present only on days the user logged a body sensation).
  { key: "body_pain", label: "Body pain" },
  // Weather metrics (present only when weather tracking is enabled; merged in
  // by entry_date before correlating).
  { key: "pressure", label: "Barometric pressure" },
  { key: "humidity", label: "Humidity" },
  { key: "temp_max", label: "Temperature" },
];

export type Strength = "weak" | "moderate" | "strong";

export interface Correlation {
  aKey: string;
  aLabel: string;
  bKey: string;
  bLabel: string;
  r: number;
  n: number;
  strength: Strength;
  direction: "positive" | "negative";
  statement: string;
}

export interface CorrelationOptions {
  minSampleSize?: number; // min paired observations
  minAbsR?: number; // min |r| to surface
  maxResults?: number;
}

function pearson(xs: number[], ys: number[]): number | null {
  const n = xs.length;
  if (n < 2) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let dx2 = 0;
  let dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  const den = Math.sqrt(dx2 * dy2);
  if (den === 0) return null; // no variance in one series
  return num / den;
}

function strengthOf(absR: number): Strength {
  if (absR >= 0.7) return "strong";
  if (absR >= 0.5) return "moderate";
  return "weak";
}

function statementFor(aLabel: string, bLabel: string, r: number, n: number, strength: Strength): string {
  const rel =
    r > 0
      ? `${aLabel} and ${bLabel} tended to rise and fall together`
      : `when ${aLabel} was higher, ${bLabel} tended to be lower`;
  return `${rel} — a ${strength} possible pattern across ${n} days. This is an association, not a cause.`;
}

/**
 * Compute the strongest correlations across the given entries.
 * Returns at most `maxResults`, sorted by |r| descending.
 */
export function computeCorrelations(
  entries: Record<string, unknown>[],
  opts: CorrelationOptions = {},
): Correlation[] {
  const minSampleSize = opts.minSampleSize ?? 8;
  const minAbsR = opts.minAbsR ?? 0.3;
  const maxResults = opts.maxResults ?? 6;

  const results: Correlation[] = [];

  for (let i = 0; i < CORRELATION_METRICS.length; i++) {
    for (let j = i + 1; j < CORRELATION_METRICS.length; j++) {
      const a = CORRELATION_METRICS[i];
      const b = CORRELATION_METRICS[j];

      const xs: number[] = [];
      const ys: number[] = [];
      for (const e of entries) {
        const va = e[a.key];
        const vb = e[b.key];
        if (typeof va === "number" && typeof vb === "number") {
          xs.push(va);
          ys.push(vb);
        }
      }

      if (xs.length < minSampleSize) continue;
      const r = pearson(xs, ys);
      if (r === null || Math.abs(r) < minAbsR) continue;

      const strength = strengthOf(Math.abs(r));
      results.push({
        aKey: a.key,
        aLabel: a.label,
        bKey: b.key,
        bLabel: b.label,
        r: Math.round(r * 100) / 100,
        n: xs.length,
        strength,
        direction: r > 0 ? "positive" : "negative",
        statement: statementFor(a.label, b.label, r, xs.length, strength),
      });
    }
  }

  return results.sort((x, y) => Math.abs(y.r) - Math.abs(x.r)).slice(0, maxResults);
}
