import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Terms of Service · MindMap",
  description: "The agreement between you and MindMap when you use the app.",
};

const EFFECTIVE_DATE = "May 10, 2026";
const VERSION = "2026-05-10";

export default function TermsPage() {
  return (
    <>
      <h1>Terms of Service</h1>
      <p className="text-sm text-muted-foreground">
        Version {VERSION} · Effective {EFFECTIVE_DATE}
      </p>

      <h2>1. Acceptance</h2>
      <p>
        By creating a MindMap account or using the MindMap app, you
        agree to these Terms and to our{" "}
        <a href="/privacy">Privacy Policy</a> and{" "}
        <a href="/medical-disclaimer">Medical Disclaimer</a>.
      </p>

      <h2>2. What MindMap is</h2>
      <p>
        MindMap is a private self-tracking journal for mood, sleep,
        routines, medication reminders, therapy notes, body sensations,
        goals, and personal wellness patterns. It is a wellness tool,
        not a medical device.
      </p>

      <h2>3. Your account</h2>
      <p>
        You are responsible for keeping your account credentials safe.
        You agree to provide accurate information at signup and to
        update it when it changes.
      </p>

      <h2>4. Acceptable use</h2>
      <p>
        Do not use MindMap to harm others, impersonate someone, or
        attempt to access another user&apos;s data. Do not reverse
        engineer the service or interfere with its operation.
      </p>

      <h2>5. Your content</h2>
      <p>
        You retain ownership of everything you enter. You grant MindMap
        a limited, non-exclusive license to store and process your
        content solely to operate the service for you.
      </p>

      <h2>6. Provider sharing</h2>
      <p>
        If you choose to share data with a provider, you authorize
        MindMap to expose only the categories, date range, and detail
        level you select, until you revoke the share.
      </p>

      <h2>7. Disclaimers</h2>
      <p>
        MindMap is provided &quot;as is&quot; without warranties of any
        kind. See our <a href="/medical-disclaimer">Medical Disclaimer</a>{" "}
        for important limitations on clinical use.
      </p>

      <h2>8. Limitation of liability</h2>
      <p>
        To the maximum extent permitted by law, MindMap and Heartwire
        are not liable for indirect, incidental, or consequential
        damages arising from your use of the app.
      </p>

      <h2>9. Termination</h2>
      <p>
        You may delete your account at any time from{" "}
        <a href="/data-deletion">Data Deletion</a>. We may suspend
        accounts that violate these Terms.
      </p>

      <h2>10. Changes</h2>
      <p>
        We may update these Terms. When we do, we will publish a new
        version and ask you to re-consent the next time you sign in.
      </p>

      <h2>11. Contact</h2>
      <p>
        For questions, contact{" "}
        <a href="mailto:support@heartwire.com">support@heartwire.com</a>.
      </p>
    </>
  );
}
