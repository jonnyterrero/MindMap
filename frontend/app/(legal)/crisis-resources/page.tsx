import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Crisis Resources · MindMap",
  description:
    "If you are in crisis, these resources can help. MindMap is not an emergency service.",
};

export default function CrisisResourcesPage() {
  return (
    <>
      <h1>Crisis Resources</h1>

      <aside
        role="note"
        aria-label="Emergency notice"
        className="not-prose my-6 rounded-lg border border-amber-300/50 bg-amber-50/60 p-4 text-sm text-amber-900 dark:border-amber-500/30 dark:bg-amber-950/30 dark:text-amber-100"
      >
        <p className="font-semibold">
          MindMap is not an emergency or crisis service.
        </p>
        <p className="mt-1 leading-relaxed">
          If you or someone else may be in danger, contact your local emergency
          services right away.
        </p>
      </aside>

      <h2>United States</h2>
      <ul>
        <li>
          <strong>988 Suicide &amp; Crisis Lifeline</strong> — call or text{" "}
          <strong>988</strong>, available 24/7.
        </li>
        <li>
          <strong>Crisis Text Line</strong> — text <strong>HOME</strong> to{" "}
          <strong>741741</strong>.
        </li>
        <li>
          <strong>Emergencies</strong> — call <strong>911</strong>.
        </li>
      </ul>

      <h2>Outside the United States</h2>
      <p>
        Find a helpline in your country at{" "}
        <a href="https://findahelpline.com" target="_blank" rel="noreferrer">
          findahelpline.com
        </a>
        . For immediate danger, contact your local emergency number.
      </p>

      <h2>A note on MindMap</h2>
      <p>
        MindMap is a private self-tracking and journaling tool for personal
        pattern discovery. It does not monitor your entries for risk and cannot
        provide crisis support. Please reach out to the resources above when you
        need help.
      </p>
    </>
  );
}
