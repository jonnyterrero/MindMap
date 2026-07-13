import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { Network } from "lucide-react";
import { getRecentMindmaps } from "./actions";
import { GraphCard } from "./graph-card";

export default async function MindmapPage() {
  const graphs = await getRecentMindmaps();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Mindmap</h1>
        <p className="text-muted-foreground">
          Concept maps built from your journal and notes. Only claims verified against your own
          words appear — unsupported ones are withheld, never invented.
        </p>
      </div>

      {graphs.length === 0 ? (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Network className="h-5 w-5 text-primary" /> No mindmaps yet
            </CardTitle>
            <CardDescription>
              Write a few journal entries — the analysis runs over them and maps the themes,
              feelings, and connections it can actually verify.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
              Your verified concept maps will appear here after the next analysis run.
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {graphs.map((row) => (
            <GraphCard key={row.id} row={row} />
          ))}
        </div>
      )}

      <MedicalDisclaimer variant="compact" />
    </div>
  );
}
