import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Support · MindMap",
  description: "How to reach the MindMap support team.",
};

export default function SupportPage() {
  return (
    <>
      <h1>Support</h1>
      <p>
        We&apos;re a small team and we read everything. Pick whichever
        channel fits.
      </p>

      <h2>Email</h2>
      <p>
        General help:{" "}
        <a href="mailto:support@heartwire.com">support@heartwire.com</a>
        <br />
        Privacy and data:{" "}
        <a href="mailto:privacy@heartwire.com">privacy@heartwire.com</a>
      </p>

      <h2>Crisis resources</h2>
      <p>
        MindMap is not a crisis service. If you are in immediate danger
        or experiencing a mental-health crisis:
      </p>
      <ul>
        <li>
          United States — call or text <strong>988</strong> for the
          Suicide &amp; Crisis Lifeline.
        </li>
        <li>
          Worldwide —{" "}
          <a
            href="https://findahelpline.com"
            target="_blank"
            rel="noreferrer"
          >
            findahelpline.com
          </a>
          .
        </li>
        <li>
          Or call your local emergency number (911, 999, 112, 000,
          etc.).
        </li>
      </ul>

      <h2>Frequently asked</h2>
      <h3>How do I export my data?</h3>
      <p>
        Sign in, open{" "}
        <a href="/settings">Settings → Export Data</a>, choose a format,
        and we&apos;ll email you a private download link.
      </p>

      <h3>How do I delete my data?</h3>
      <p>
        See <a href="/data-deletion">Data Deletion</a>.
      </p>

      <h3>Does MindMap diagnose conditions?</h3>
      <p>
        No. See <a href="/medical-disclaimer">Medical Disclaimer</a>.
      </p>
    </>
  );
}
