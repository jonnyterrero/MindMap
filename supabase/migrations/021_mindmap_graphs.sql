-- ============================================================
-- Migration 021: verified mindmap graphs
-- The Python graph pipeline (ml/mindmap_ml/graph/) WRITES one verified
-- MindmapArtifact per source document; the Next app only READS it.
-- Additive & idempotent. Writes are service-role only (no user insert/update
-- policy); users read their own, providers read when granted read_reports
-- (mirrors migrations 017/020).
-- ============================================================

BEGIN;

CREATE TABLE IF NOT EXISTS public.mindmap_graphs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  doc_id text NOT NULL,                               -- Stage-1 document id (content hash)
  mindmap_id text NOT NULL,
  source_type text NOT NULL DEFAULT 'journal',        -- journal | brainstorm | notes | voice_transcript
  abstained boolean NOT NULL DEFAULT false,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,         -- MindmapArtifact.to_dict()
  pipeline_version text NOT NULL DEFAULT 'graph_pipeline_v0',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mindmap_graphs_user_time
  ON public.mindmap_graphs (user_id, created_at DESC);

-- Idempotent batch upsert: one artifact per (user, source doc, pipeline version).
CREATE UNIQUE INDEX IF NOT EXISTS uq_mindmap_graphs_doc
  ON public.mindmap_graphs (user_id, doc_id, pipeline_version);

ALTER TABLE public.mindmap_graphs ENABLE ROW LEVEL SECURITY;

-- Patient reads their own graphs.
DO $$ BEGIN
  CREATE POLICY mindmap_graphs_select_own ON public.mindmap_graphs
    FOR SELECT USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Provider reads when granted read_reports (a verified graph is report-like).
DO $$ BEGIN
  CREATE POLICY mindmap_graphs_provider_read ON public.mindmap_graphs
    FOR SELECT USING (
      EXISTS (
        SELECT 1 FROM public.mindmap_provider_access pa
        WHERE pa.provider_user_id = auth.uid()
          AND pa.patient_user_id = mindmap_graphs.user_id
          AND pa.revoked_at IS NULL
          AND (pa.permissions->>'read_reports')::boolean IS TRUE
      )
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMIT;
