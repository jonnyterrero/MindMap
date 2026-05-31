import { getJournalEntries, getAiReflectionEnabled, getJournalAnalyses } from "./actions";
import { JournalList } from "./journal-list";

export default async function JournalPage() {
  const [entries, aiEnabled, analyses] = await Promise.all([
    getJournalEntries(),
    getAiReflectionEnabled(),
    getJournalAnalyses(),
  ]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Journal</h1>
        <p className="text-muted-foreground">
          Write freely about your day, thoughts, and experiences.
        </p>
      </div>
      <JournalList entries={entries} aiEnabled={aiEnabled} analyses={analyses} />
    </div>
  );
}
