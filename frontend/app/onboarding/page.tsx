import { redirect } from "next/navigation";
import { checkOnboardingStatus } from "./actions";
import { OnboardingFlow } from "./onboarding-flow";

export default async function OnboardingPage() {
  const done = await checkOnboardingStatus();
  if (done) redirect("/today");

  return <OnboardingFlow />;
}
