import { createClient } from "@/lib/supabase-server";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BodyMapView } from "./body-map-view";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";

export const dynamic = "force-dynamic";

type SensationRow = {
  id: string;
  body_part: string;
  intensity: number;
  notes: string | null;
  created_at: string;
  entry_id: string;
};

export default async function BodyMapPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return null;
  }

  // RLS scopes this to the calling user via entry_id → mindmap_entries.user_id.
  const sinceIso = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();

  const { data: sensations } = await supabase
    .from("mindmap_body_sensations")
    .select("id, body_part, intensity, notes, created_at, entry_id")
    .gte("created_at", sinceIso)
    .order("created_at", { ascending: false })
    .limit(500);

  const rows = (sensations ?? []) as SensationRow[];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Body map</h1>
        <p className="text-sm text-muted-foreground">
          Where you&apos;ve logged sensations over the last 30 days.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Heatmap</CardTitle>
        </CardHeader>
        <CardContent>
          <BodyMapView sensations={rows} />
        </CardContent>
      </Card>

      <MedicalDisclaimer variant="inline" />
    </div>
  );
}
