import { getTherapySessions } from "./actions";
import { TherapyList } from "./therapy-list";

export default async function TherapyPage() {
  const sessions = await getTherapySessions();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Therapy Sessions</h1>
        <p className="text-muted-foreground">
          Log your therapy sessions and track mood changes.
        </p>
      </div>
      <TherapyList sessions={sessions} />
    </div>
  );
}
