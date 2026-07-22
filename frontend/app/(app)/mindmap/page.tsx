import { getMindmaps } from "./actions";
import { MindmapView } from "./mindmap-view";

export const metadata = {
  title: "Mindmap",
};

export default async function MindmapPage() {
  const rows = await getMindmaps();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Mindmap</h1>
        <p className="text-muted-foreground">
          A verified concept map built from your journal entries. Every concept is
          grounded in what you wrote — nothing here is a diagnosis or advice.
        </p>
      </div>
      <MindmapView rows={rows} />
    </div>
  );
}
