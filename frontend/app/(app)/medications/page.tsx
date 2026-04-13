import { getMedicationSchedules } from "./actions";
import { MedicationsList } from "./medications-list";

export default async function MedicationsPage() {
  const schedules = await getMedicationSchedules();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Medications</h1>
        <p className="text-muted-foreground">
          Manage your medication schedule. Active medications appear on your
          Today page for daily tracking.
        </p>
      </div>
      <MedicationsList schedules={schedules} />
    </div>
  );
}
