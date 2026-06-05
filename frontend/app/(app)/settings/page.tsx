import { getProfile } from "./actions";
import { SettingsForm } from "./settings-form";
import { WeatherSettings } from "./weather-settings";
import { AiSettings } from "./ai-settings";
import { ProviderSharing } from "./provider-sharing";
import { DataPrivacy } from "./data-privacy";
import { getMyRole, getMyGrants } from "@/app/(app)/provider/actions";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";

export default async function SettingsPage() {
  const [profile, role, grants] = await Promise.all([
    getProfile(),
    getMyRole(),
    getMyGrants(),
  ]);

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
      <AiSettings enabled={Boolean(profile?.ai_reflection_enabled)} />
      <ProviderSharing grants={grants} isProvider={role === "provider"} />
      <DataPrivacy />
      <MedicalDisclaimer variant="full" />
    </div>
  );
}
