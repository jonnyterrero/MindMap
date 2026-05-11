import type { Metadata } from "next";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";

export const metadata: Metadata = {
  title: "Medical Disclaimer · MindMap",
  description:
    "MindMap is for self-tracking and wellness reflection only. Not a medical device.",
};

const EFFECTIVE_DATE = "May 10, 2026";
const VERSION = "2026-05-10";

export default function MedicalDisclaimerPage() {
  return (
    <>
      <h1>Medical Disclaimer</h1>
      <p className="text-sm text-muted-foreground">
        Version {VERSION} · Effective {EFFECTIVE_DATE}
      </p>

      <MedicalDisclaimer variant="full" className="not-prose my-6" />

      <h2>What MindMap is</h2>
      <p>
        MindMap is a private journal and self-tracking tool. It helps
        you observe patterns across mood, sleep, medication adherence,
        routines, body sensations, and other personal wellness signals
        that you choose to log.
      </p>

      <h2>What MindMap is not</h2>
      <ul>
        <li>Not a medical device.</li>
        <li>
          Not a diagnostic tool for depression, anxiety, bipolar
          disorder, migraines, or any other condition.
        </li>
        <li>Not a substitute for evaluation or treatment by a clinician.</li>
        <li>Not an emergency or crisis service.</li>
      </ul>

      <h2>How to read in-app insights</h2>
      <p>
        Insights surfaced inside MindMap are based on your own logged
        data. They describe possible patterns and correlations only —
        for example, &quot;your sleep averaged less than six hours this
        week.&quot; They do not constitute medical advice. Always
        discuss health decisions with a qualified clinician.
      </p>

      <h2>In a crisis</h2>
      <p>
        If you are experiencing a medical emergency or mental-health
        crisis, contact emergency services or a local crisis hotline
        immediately. In the United States you can call or text{" "}
        <strong>988</strong> for the Suicide &amp; Crisis Lifeline.
        Outside the US, see{" "}
        <a
          href="https://findahelpline.com"
          target="_blank"
          rel="noreferrer"
        >
          findahelpline.com
        </a>
        .
      </p>
    </>
  );
}
