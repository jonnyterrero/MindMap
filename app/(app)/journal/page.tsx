import { getJournalEntries } from "./actions";
import { JournalList } from "./journal-list";

export default async function JournalPage() {
  const entries = await getJournalEntries();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Journal</h1>
        <p className="text-muted-foreground">
          Write freely about your day, thoughts, and experiences.
        </p>
      </div>
      <JournalList entries={entries} />
    </div>
  );
}
