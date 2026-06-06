import Link from "next/link";
import { createClient } from "@/lib/supabase-server";
import { getMyRole, getMyPatients } from "./actions";
import { PatientSummaryCard } from "@/components/provider/patient-summary-card";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Stethoscope, Users, ArrowLeft } from "lucide-react";

export default async function ProviderDashboardPage() {
  const role = await getMyRole();

  if (role !== "provider") {
    return (
      <div className="mx-auto max-w-lg">
        <Card>
          <CardHeader className="text-center">
            <div className="mb-2 flex justify-center">
              <div className="rounded-full bg-muted p-3">
                <Stethoscope className="h-7 w-7 text-muted-foreground" />
              </div>
            </div>
            <CardTitle>Provider area</CardTitle>
            <CardDescription>
              This dashboard is for clinicians and coaches. If you&apos;re a patient, you can
              share your data with a provider from Settings.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex justify-center">
            <Button asChild variant="outline">
              <Link href="/home"><ArrowLeft /> Back to MindMap</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  const patients = await getMyPatients();

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Provider dashboard</h1>
        <p className="text-muted-foreground">Read-only summaries for patients who shared their data.</p>
      </div>

      {patients.length === 0 ? (
        <Card>
          <CardHeader className="text-center">
            <div className="mb-2 flex justify-center">
              <div className="rounded-full bg-primary/10 p-3">
                <Users className="h-7 w-7 text-primary" />
              </div>
            </div>
            <CardTitle>No patients yet</CardTitle>
            <CardDescription>Share your provider code with patients to get started.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-lg border bg-muted/40 p-3 text-center">
              <p className="text-xs text-muted-foreground">Your provider code</p>
              <p className="mt-1 select-all break-all font-mono text-sm">{user?.id}</p>
            </div>
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="space-y-3">
            {patients.map((p) => (
              <PatientSummaryCard key={p.patientId} patient={p} />
            ))}
          </div>
          <div className="rounded-lg border bg-muted/40 p-3 text-center">
            <p className="text-xs text-muted-foreground">Your provider code (share to add patients)</p>
            <p className="mt-1 select-all break-all font-mono text-xs">{user?.id}</p>
          </div>
        </>
      )}
    </div>
  );
}
