"use server";

import { createClient } from "@/lib/supabase-server";

/**
 * Read-only access to the verified concept graphs written by the Python graph
 * pipeline (`mindmap_graphs`). The app never generates or verifies claims —
 * it only reads fail-closed artifacts for the signed-in user (RLS-scoped).
 */

export type GraphConfidence = {
  raw_score: number;
  calibrated: number;
  bucket: "low" | "medium" | "high" | string;
  calibrator_version: string;
};

export type GraphNode = {
  node_id: string;
  label: string;
  node_type: string;
  evidence: string[];
  claim_class: string;
  status: string;
  confidence: GraphConfidence | null;
};

export type GraphEdge = {
  edge_id: string;
  src: string;
  dst: string;
  edge_type: string;
  evidence: string[];
  claim_class: string;
  status: string;
  confidence: GraphConfidence | null;
};

export type MindmapPayload = {
  mindmap_id: string;
  doc_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  suppressed: { claim_id: string; kind: string; reason_codes: string[] }[];
  abstained: boolean;
  evidence_texts?: Record<string, string>;
  source_meta?: {
    title?: string;
    entry_date?: string;
    journal_entry_id?: string;
    obsidian_path?: string;
    tags?: string[];
    wikilinks?: string[];
  };
};

export type MindmapGraphRow = {
  id: string;
  doc_id: string;
  source_type: string;
  abstained: boolean;
  payload: MindmapPayload;
  pipeline_version: string;
  created_at: string;
};

export async function getRecentMindmaps(limit = 10): Promise<MindmapGraphRow[]> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return [];

  const { data } = await supabase
    .from("mindmap_graphs")
    .select("id, doc_id, source_type, abstained, payload, pipeline_version, created_at")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false })
    .limit(limit);

  return (data as MindmapGraphRow[] | null) ?? [];
}
