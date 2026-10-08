import {
  getTodayEntry,
  getActiveRoutinesWithStatus,
  getCheckinConfig,
  getCheckInCount,
} from "./actions";
import { getTodayAdherence } from "@/app/(app)/medications/actions";
import { GuidedCheckin } from "./guided-checkin";
import { createClient } from "@/lib/supabase-server";
import { formatCalendarHeading } from "@/lib/local-date";
import { userTimeZone } from "@/lib/user-local-today";

export default async function TodayPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  const heading = formatCalendarHeading(user ? await userTimeZone(supabase, user.id) : null);

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
        <h1 className="text-2xl font-bold tracking-tight">{heading}</h1>
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
