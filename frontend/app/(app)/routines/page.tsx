import { getRoutines } from "./actions";
import { RoutinesList } from "./routines-list";

export default async function RoutinesPage() {
  const routines = await getRoutines();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Routines</h1>
        <p className="text-muted-foreground">
          Manage your daily routines. Active routines show up on your Today page.
        </p>
      </div>
      <RoutinesList routines={routines} />
    </div>
  );
}
