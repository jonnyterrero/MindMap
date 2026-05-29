import { getProfile } from "./actions";
import { SettingsForm } from "./settings-form";
import { WeatherSettings } from "./weather-settings";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";

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
      <WeatherSettings
        enabled={Boolean(profile?.weather_enabled)}
        label={(profile?.weather_label as string | null) ?? null}
      />
      <MedicalDisclaimer variant="full" />
    </div>
  );
}
