import { getTodayEntry, getActiveRoutinesWithStatus, getBodySensations } from "./actions";
import { getTodayAdherence } from "@/app/(app)/medications/actions";
import { TodayForm } from "./today-form";
import { RoutineChecklist } from "./routine-checklist";
import { MedChecklist } from "./med-checklist";
import { BodySensations } from "./body-sensations";

export default async function TodayPage() {
  const [entry, routines, meds] = await Promise.all([
    getTodayEntry(),
    getActiveRoutinesWithStatus(),
    getTodayAdherence(),
  ]);

  const sensations = entry ? await getBodySensations(entry.id) : [];

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
      {meds.length > 0 && <MedChecklist meds={meds} />}
      {routines.length > 0 && <RoutineChecklist routines={routines} />}
      <TodayForm initialData={entry} />
      {entry && <BodySensations sensations={sensations} />}
    </div>
  );
}
