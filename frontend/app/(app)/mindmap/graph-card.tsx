import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, Quote, ShieldCheck } from "lucide-react";
import type { GraphNode, MindmapGraphRow } from "./actions";

const BUCKET_STYLES: Record<string, string> = {
  high: "border-emerald-500/50 text-emerald-600 dark:text-emerald-400",
  medium: "border-amber-500/50 text-amber-600 dark:text-amber-400",
  low: "border-muted-foreground/40 text-muted-foreground",
};

function ConfidenceBadge({ node }: { node: GraphNode }) {
  const bucket = node.confidence?.bucket ?? "low";
  return (
    <Badge variant="outline" className={BUCKET_STYLES[bucket] ?? BUCKET_STYLES.low}>
      {node.label}
      <span className="ml-1 opacity-70">{bucket}</span>
    </Badge>
  );
}

function cardTitle(row: MindmapGraphRow): string {
  const meta = row.payload.source_meta;
  if (meta?.title) return meta.title;
  if (meta?.obsidian_path) return meta.obsidian_path;
  if (meta?.entry_date) return `Journal · ${meta.entry_date}`;
  return `Mindmap · ${new Date(row.created_at).toLocaleDateString()}`;
}

export function GraphCard({ row }: { row: MindmapGraphRow }) {
  const p = row.payload;
  const quotes = p.evidence_texts ?? {};
  const nodesByType = new Map<string, GraphNode[]>();
  for (const n of p.nodes) {
    nodesByType.set(n.node_type, [...(nodesByType.get(n.node_type) ?? []), n]);
  }
  const labelById = new Map(p.nodes.map((n) => [n.node_id, n.label]));

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <ShieldCheck className="h-4 w-4 text-primary" />
          {cardTitle(row)}
        </CardTitle>
        <CardDescription>
          {row.source_type} · {new Date(row.created_at).toLocaleDateString()} · every claim below
          passed verification against your own words
          {p.suppressed.length > 0 && ` · ${p.suppressed.length} unsupported claim${p.suppressed.length === 1 ? "" : "s"} withheld`}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {p.abstained || p.nodes.length === 0 ? (
          <div className="rounded-lg border border-dashed p-4 text-center text-sm text-muted-foreground">
            Nothing could be verified from this entry, so nothing is shown — the pipeline never
            guesses.
          </div>
        ) : (
          <>
            {[...nodesByType.entries()].map(([type, nodes]) => (
              <section key={type}>
                <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {type}s
                </h3>
                <div className="flex flex-wrap gap-1.5">
                  {nodes.map((n) => (
                    <ConfidenceBadge key={n.node_id} node={n} />
                  ))}
                </div>
              </section>
            ))}

            {p.edges.length > 0 && (
              <section>
                <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Connections
                </h3>
                <ul className="space-y-2 text-sm">
                  {p.edges.map((e) => {
                    const quote = e.evidence.map((sid) => quotes[sid]).find(Boolean);
                    return (
                      <li key={e.edge_id} className="rounded-lg border p-2.5">
                        <span className="flex flex-wrap items-center gap-1.5">
                          <span className="font-medium">{labelById.get(e.src) ?? e.src}</span>
                          <ArrowRight className="h-3.5 w-3.5 text-muted-foreground" />
                          <span className="font-medium">{labelById.get(e.dst) ?? e.dst}</span>
                          <Badge variant="secondary" className="text-xs">
                            {e.edge_type}
                          </Badge>
                        </span>
                        {quote && (
                          <p className="mt-1 flex items-start gap-1 text-xs text-muted-foreground">
                            <Quote className="mt-0.5 h-3 w-3 shrink-0" />
                            “{quote}”
                          </p>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </section>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
