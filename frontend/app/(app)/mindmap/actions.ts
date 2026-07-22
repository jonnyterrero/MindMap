"use server";

import { createClient } from "@/lib/supabase-server";

/**
 * Read-only access to the verified mindmaps written by the Python batch
 * (`mindmap_graphs`). The app never builds these — it only reads the verified
 * concept graph the pipeline extracted from the user's own journal text
 * (RLS-scoped to the signed-in user). Every node/edge that appears here has
 * passed Stage-3 verification; nothing is a diagnosis or clinical claim.
 */

export type GraphConfidence = {
  raw_score: number;
  calibrated: number;
  bucket: "low" | "medium" | "high";
  calibrator_version: string;
};

export type GraphNode = {
  node_id: string;
  label: string;
  node_type: string;
  evidence: string[];
  claim_class: "directly_supported" | "weakly_inferred" | "unverifiable";
  status: string;
  confidence: GraphConfidence | null;
};

export type GraphEdge = {
  edge_id: string;
  src: string;
  dst: string;
  edge_type: string;
  claim_class: "directly_supported" | "weakly_inferred" | "unverifiable";
  status: string;
  confidence: GraphConfidence | null;
};

export type MindmapPayload = {
  mindmap_id: string;
  doc_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  coverage: { spans_total: number; spans_used: number; ratio: number };
  abstained: boolean;
  created_at: string;
};

export type MindmapRow = {
  id: string;
  source_id: string;
  entry_date: string | null;
  abstained: boolean;
  payload: MindmapPayload;
  pipeline_version: string;
  updated_at: string;
};

/**
 * Latest verified mindmaps for the signed-in user, newest entry first.
 * `limit` keeps the page bounded; the map view paginates client-side if needed.
 */
export async function getMindmaps(limit = 30): Promise<MindmapRow[]> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return [];

  const { data } = await supabase
    .from("mindmap_graphs")
    .select("id, source_id, entry_date, abstained, payload, pipeline_version, updated_at")
    .eq("user_id", user.id)
    .order("entry_date", { ascending: false, nullsFirst: false })
    .order("updated_at", { ascending: false })
    .limit(limit);

  return (data as MindmapRow[] | null) ?? [];
}
