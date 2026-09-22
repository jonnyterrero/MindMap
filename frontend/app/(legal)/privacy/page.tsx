import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy · MindMap",
  description:
    "How MindMap collects, uses, and protects your wellness data.",
};

const EFFECTIVE_DATE = "May 10, 2026";
const VERSION = "2026-05-10";

export default function PrivacyPage() {
  return (
    <>
      <h1>Privacy Policy</h1>
      <p className="text-sm text-muted-foreground">
        Version {VERSION} · Effective {EFFECTIVE_DATE}
      </p>

      <h2>1. What data we collect</h2>
      <p>
        MindMap collects information you explicitly enter, including daily
        wellness check-ins (mood, sleep, anxiety, depression, mania, focus,
        productivity, HRV, migraines, body sensations, notes), journal
        entries, medication schedules and adherence, routines, goals,
        therapy session records, reminders, and triggers. We also collect
        your account email, authentication identifiers, and basic
        diagnostic logs.
      </p>

      <h2>2. Why we collect it</h2>
      <p>
        Your data powers the in-app dashboards, trends, reminders, and
        self-tracking insights. We do not sell your data. We do not use it
        for advertising.
      </p>

      <h2>3. How we use it</h2>
      <p>
        Data is processed to (a) render your own dashboards and exports,
        (b) compute pattern-detection insights you can review, (c) send
        the notifications you have opted into, and (d) fulfill your
        explicit sharing choices.
      </p>

      <h2>4. Sharing with providers</h2>
      <p>
        Sharing your data with a clinician, coach, or supporter is
        opt-in. You choose which categories, what date range, and what
        detail level. You can revoke any share at any time from{" "}
        <a href="/settings">Settings</a>. Every provider read is logged
        in your audit trail.
      </p>

      <h2>5. Analytics &amp; crash reporting</h2>
      <p>
        Optional product analytics and performance measurement are off by
        default. You can enable or disable them under Settings → Privacy. When
        enabled, we use Vercel Analytics, Vercel Speed Insights, and PostHog
        with a pseudonymous account identifier, but without journal text,
        symptom values, or other personal health data.
        Separate operational error monitoring may receive de-identified
        technical diagnostics so we can identify crashes; session replay and
        default PII collection are disabled.
      </p>

      <h2>6. Push notifications</h2>
      <p>
        Push notifications require your explicit permission on iOS,
        Android, and the web. Notification content is limited to the
        reminders you have configured.
      </p>

      <h2>7. Data export</h2>
      <p>
        You can request an export of all your data at any time from{" "}
        <a href="/settings">Settings</a>. Exports are produced in CSV,
        JSON, or PDF and made available through a private, expiring
        download link.
      </p>

      <h2>8. Data deletion</h2>
      <p>
        You can request deletion of your account or specific data
        categories at any time from{" "}
        <a href="/data-deletion">Data Deletion</a>. Soft-deletion is
        immediate; hard-deletion follows our retention window.
      </p>

      <h2>9. Security</h2>
      <p>
        Data is stored in Supabase Postgres with row-level security
        enforced on every health-related table. All traffic is encrypted
        in transit (TLS 1.2+). Journal entries you write are encrypted at
        the application level using envelope encryption (AES-256-GCM):
        each account has its own data-encryption key, wrapped by a master
        key we hold on the server. This protects the journal body in
        database backups and dumps; it does not hide journal content from
        our own systems when features you have opted into (AI reflection,
        AI reports) send that content to our AI provider on your behalf.
      </p>

      <h2>10. Children</h2>
      <p>
        MindMap is not directed at children under 13 and we do not
        knowingly collect data from them.
      </p>

      <h2>11. Contact</h2>
      <p>
        For privacy questions, contact{" "}
        <a href="mailto:privacy@heartwire.com">privacy@heartwire.com</a>.
      </p>
    </>
  );
}
