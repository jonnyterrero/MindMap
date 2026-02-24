import { getTodayEntry, getActiveRoutinesWithStatus } from "./actions";
import { TodayForm } from "./today-form";
import { RoutineChecklist } from "./routine-checklist";

export default async function TodayPage() {
  const [entry, routines] = await Promise.all([
    getTodayEntry(),
    getActiveRoutinesWithStatus(),
  ]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">
          {new Date().toLocaleDateString("en-US", {
            weekday: "long",
            month: "long",
            day: "numeric",
          })}
        </h1>
        <p className="text-muted-foreground">
          {entry ? "Your entry for today — update anytime." : "How are you doing today?"}
        </p>
      </div>
      {routines.length > 0 && <RoutineChecklist routines={routines} />}
      <TodayForm initialData={entry} />
    </div>
  );
}
