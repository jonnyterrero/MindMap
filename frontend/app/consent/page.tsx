import { checkConsentStatus } from "./actions";
import { redirect } from "next/navigation";
import { ConsentForm } from "./consent-form";

export default async function ConsentPage() {
  const hasConsented = await checkConsentStatus();
  if (hasConsented) redirect("/today");

  return (
    <div className="min-h-[80vh] flex items-center justify-center px-4">
      <div className="max-w-lg w-full">
        <ConsentForm />
      </div>
    </div>
  );
}
