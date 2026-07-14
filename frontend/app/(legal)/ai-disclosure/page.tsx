import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI / ML Disclosure · MindMap",
  description:
    "How MindMap uses machine learning to surface possible patterns from the data you log.",
};

const EFFECTIVE_DATE = "May 10, 2026";
const VERSION = "2026-05-10";

export default function AiDisclosurePage() {
  return (
    <>
      <h1>AI / ML Disclosure</h1>
      <p className="text-sm text-muted-foreground">
        Version {VERSION} · Effective {EFFECTIVE_DATE}
      </p>

      <h2>What this covers</h2>
      <p>
        MindMap uses statistical and machine-learning techniques to highlight
        possible patterns and correlations in the data you choose to log. This
        page explains, in plain language, what those features do and do not do.
      </p>

      <h2>What the models do</h2>
      <ul>
        <li>
          Compute trends and correlations across your own entries — for
          example, relating logged sleep to logged mood.
        </li>
        <li>
          Summarize your history into plain-language reflections and charts.
        </li>
        <li>
          Surface possible patterns for you to explore and discuss with a
          qualified professional.
        </li>
      </ul>

      <h2>What the models do not do</h2>
      <ul>
        <li>They do not diagnose, treat, cure, or prevent any condition.</li>
        <li>
          They do not predict illness or make clinical or medication
          recommendations.
        </li>
        <li>
          They do not infer information about you from outside sources — only
          the data you record is used.
        </li>
      </ul>

      <h2>Accuracy and limitations</h2>
      <p>
        Insights are based only on the data you choose to log and may be
        incomplete or inaccurate. Correlation does not imply causation. Always
        use your own judgment and consult a qualified clinician for health
        decisions.
      </p>

      <h2>Your control</h2>
      <p>
        You decide what to track. You can export or delete your data at any
        time — see our{" "}
        <a href="/data-deletion">Data Deletion</a> page for details.
      </p>
    </>
  );
}
