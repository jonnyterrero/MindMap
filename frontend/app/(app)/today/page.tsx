import {
  getTodayEntry,
  getActiveRoutinesWithStatus,
  getCheckinConfig,
  getCheckInCount,
} from "./actions";
import { getTodayAdherence } from "@/app/(app)/medications/actions";
import { GuidedCheckin } from "./guided-checkin";

export default async function TodayPage() {
  const [entry, routines, meds, config, checkInsCompleted] = await Promise.all([
    getTodayEntry(),
    getActiveRoutinesWithStatus(),
    getTodayAdherence(),
    getCheckinConfig(),
    getCheckInCount(),
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
          {entry ? "Your check-in for today — update anytime." : "How are you doing today?"}
        </p>
      </div>

      <GuidedCheckin
        initialEntry={entry}
        routines={routines}
        meds={meds}
        cards={config.cards}
        checkInsCompleted={checkInsCompleted}
      />
    </div>
  );
}
