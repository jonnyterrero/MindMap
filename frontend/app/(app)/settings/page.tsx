import { getProfile } from "./actions";
import { SettingsForm } from "./settings-form";

export default async function SettingsPage() {
  const profile = await getProfile();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">
          Manage your profile and account preferences.
        </p>
      </div>
      <SettingsForm profile={profile} />
    </div>
  );
}
