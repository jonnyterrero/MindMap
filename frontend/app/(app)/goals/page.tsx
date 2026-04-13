import { getGoals } from "./actions";
import { GoalsList } from "./goals-list";

export default async function GoalsPage() {
  const goals = await getGoals();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Goals</h1>
        <p className="text-muted-foreground">
          Set and track your mental health and wellness goals.
        </p>
      </div>
      <GoalsList goals={goals} />
    </div>
  );
}
