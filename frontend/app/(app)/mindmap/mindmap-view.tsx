import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ArrowRight,
  Flag,
  HelpCircle,
  Heart,
  Lightbulb,
  Layers,
  CalendarClock,
  Tag,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { GraphEdge, GraphNode, MindmapRow } from "./actions";

// node_type -> presentation (icon + human label). Mirrors graph/schema.py NODE_TYPES.
const NODE_META: Record<string, { icon: LucideIcon; label: string }> = {
  theme: { icon: Layers, label: "Themes" },
  emotion: { icon: Heart, label: "Emotions" },
  event: { icon: CalendarClock, label: "Events" },
  goal: { icon: Flag, label: "Goals" },
  value: { icon: Sparkles, label: "Values" },
  entity: { icon: Tag, label: "People & things" },
  question: { icon: HelpCircle, label: "Open questions" },
};

const NODE_ORDER = ["theme", "emotion", "event", "goal", "value", "entity", "question"];

// edge_type -> connective phrase for the relationship list.
const EDGE_PHRASE: Record<string, string> = {
  causal: "may relate to",
  temporal: "then",
  thematic: "connects to",
  contrast: "contrasts with",
  elaboration: "expands on",
  part_of: "is part of",
};

function ClaimBadge({ claimClass }: { claimClass: GraphNode["claim_class"] }) {
  // Honest provenance: "Stated" = grounded in the text; "Inferred" = a weaker,
  // reflected connection. Never presented as fact/diagnosis.
  const inferred = claimClass !== "directly_supported";
  return (
    <span
      className={cn(
        "ml-2 shrink-0 rounded-full px-1.5 py-0.5 text-[10px] font-medium",
        inferred
          ? "border border-border text-muted-foreground"
          : "bg-primary/15 text-primary",
      )}
    >
      {inferred ? "Inferred" : "Stated"}
    </span>
  );
}

function NodeGroup({ type, nodes }: { type: string; nodes: GraphNode[] }) {
  const meta = NODE_META[type] ?? { icon: Tag, label: type };
  const Icon = meta.icon;
  return (
    <div>
      <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        <Icon className="h-3.5 w-3.5" aria-hidden="true" />
        {meta.label}
      </div>
      <ul className="flex flex-wrap gap-1.5">
        {nodes.map((n) => (
          <li
            key={n.node_id}
            className="flex items-center rounded-lg border border-border/70 bg-card/60 px-2.5 py-1 text-sm"
          >
            <span>{n.label}</span>
            <ClaimBadge claimClass={n.claim_class} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function Relationships({
  edges,
  labelOf,
}: {
  edges: GraphEdge[];
  labelOf: (id: string) => string;
}) {
  if (edges.length === 0) return null;
  return (
    <div className="mt-4 border-t border-border/60 pt-3">
      <div className="mb-1.5 text-xs font-medium text-muted-foreground">
        Connections
      </div>
      <ul className="space-y-1">
        {edges.map((e) => (
          <li key={e.edge_id} className="flex flex-wrap items-center gap-1.5 text-sm">
            <span className="font-medium">{labelOf(e.src)}</span>
            <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
              <ArrowRight className="h-3 w-3" aria-hidden="true" />
              {EDGE_PHRASE[e.edge_type] ?? "relates to"}
            </span>
            <span className="font-medium">{labelOf(e.dst)}</span>
            {e.claim_class !== "directly_supported" && (
              <span className="rounded-full border border-border px-1.5 py-0.5 text-[10px] text-muted-foreground">
                Inferred
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function formatDate(entryDate: string | null, updatedAt: string): string {
  const raw = entryDate ?? updatedAt;
  const d = new Date(entryDate ? `${raw}T00:00:00` : raw);
  return Number.isNaN(d.getTime())
    ? "Journal entry"
    : d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric", year: "numeric" });
}

function MindmapCard({ row }: { row: MindmapRow }) {
  const { payload } = row;
  const labelOf = (id: string) =>
    payload.nodes.find((n) => n.node_id === id)?.label ?? "…";

  const grouped = NODE_ORDER.map((type) => ({
    type,
    nodes: payload.nodes.filter((n) => n.node_type === type),
  })).filter((g) => g.nodes.length > 0);

  const mapped = payload.coverage?.spans_used ?? 0;
  const total = payload.coverage?.spans_total ?? 0;

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="text-base">{formatDate(row.entry_date, row.updated_at)}</CardTitle>
          {total > 0 && (
            <span className="text-xs text-muted-foreground">
              {mapped} of {total} passages mapped
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {row.abstained || payload.nodes.length === 0 ? (
          <p className="rounded-lg border border-dashed p-4 text-center text-sm text-muted-foreground">
            Not enough clear structure in this entry to map yet.
          </p>
        ) : (
          <>
            <div className="grid gap-4 sm:grid-cols-2">
              {grouped.map((g) => (
                <NodeGroup key={g.type} type={g.type} nodes={g.nodes} />
              ))}
            </div>
            <Relationships edges={payload.edges} labelOf={labelOf} />
          </>
        )}
      </CardContent>
    </Card>
  );
}

export function MindmapView({ rows }: { rows: MindmapRow[] }) {
  if (rows.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Lightbulb className="h-5 w-5 text-primary" aria-hidden="true" /> Your Mindmap
          </CardTitle>
          <CardDescription>
            A verified concept map built from your journal entries. Not a diagnosis.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
            Write a journal entry and your mindmap will appear here after the daily
            analysis runs. Every concept shown is grounded in what you wrote.
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {rows.map((row) => (
        <MindmapCard key={row.id} row={row} />
      ))}
    </div>
  );
}
