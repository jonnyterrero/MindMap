import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Data Deletion · MindMap",
  description:
    "How to request deletion of your MindMap account and data.",
};

export default function DataDeletionPage() {
  return (
    <>
      <h1>Data Deletion</h1>
      <p className="text-sm text-muted-foreground">
        You can delete your MindMap account and data at any time.
      </p>

      <h2>From inside the app</h2>
      <ol>
        <li>
          Sign in and open{" "}
          <a href="/settings">Settings → Privacy &amp; Data</a>.
        </li>
        <li>
          Choose <strong>Request Account Deletion</strong>, or pick a
          narrower scope (entries, journal, medications, routines,
          therapy sessions, goals, or exports).
        </li>
        <li>
          Confirm the request from the confirmation email we send to
          your account address.
        </li>
        <li>
          We soft-delete immediately and hard-delete after our standard
          retention window. Provider shares are revoked at submission
          time.
        </li>
      </ol>

      <h2>By email</h2>
      <p>
        If you cannot access your account, email{" "}
        <a href="mailto:privacy@heartwire.com">privacy@heartwire.com</a>{" "}
        from the address associated with your MindMap account. Include
        the words &quot;Delete my MindMap data&quot; in the subject. We
        respond within 30 days.
      </p>

      <h2>What gets deleted</h2>
      <p>
        Full account deletion removes daily entries, journal, body
        sensations, medication schedules and adherence, routines, goals,
        therapy sessions, reminders, triggers, insights, exports,
        consent records, push tokens, and provider sharing
        configuration. Audit log entries are retained in
        anonymised form for our compliance window.
      </p>

      <h2>App Store / Google Play</h2>
      <p>
        This page is the official data-deletion mechanism referenced in
        our App Store and Google Play listings. No login is required to
        reach it.
      </p>
    </>
  );
}
