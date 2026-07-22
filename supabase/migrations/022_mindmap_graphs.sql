-- ============================================================
-- Migration 022: Verified mindmap graphs
-- The Python batch (ml/mindmap_ml/serving/graph_batch.py) runs the
-- text->verified-mindmap pipeline over journal entries and WRITES one
-- artifact row per (user, journal entry, pipeline version); the Next app
-- only READS them. Additive & idempotent, mirroring migration 020.
--
-- Access model (narrower than 020 ON PURPOSE): the payload quotes journal
-- text verbatim (evidence spans), and mindmap_journal_entries is user-own
-- only -- providers have no journal permission key (017 grants only
-- read_predictions / read_reports / read_entries). Derived text must not be
-- readable more broadly than its source, so there is NO provider policy
-- here. Writes are service-role only (no insert/update policy for users).
--
-- NOTE: prod reached this shape in two steps (an initial doc_id-keyed sketch,
-- then journal linkage + provider-policy removal). This file is the flattened
-- end state so a fresh database matches prod from one migration.
-- ============================================================

BEGIN;

CREATE TABLE IF NOT EXISTS public.mindmap_graphs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  -- pipeline identifiers (from MindmapArtifact)
  doc_id text NOT NULL,                               -- Stage-1 document id (content-derived)
  mindmap_id text NOT NULL,                           -- artifact id ("mm_<doc_id>")
  source_type text NOT NULL DEFAULT 'journal',
  -- source linkage (idempotency key)
  source_table text NOT NULL DEFAULT 'mindmap_journal_entries',
  source_id uuid NOT NULL,                            -- journal entry id
  entry_date date,
  content_sha text NOT NULL DEFAULT '',              -- sha of source text (skip-unchanged)
  -- verified output
  abstained boolean NOT NULL DEFAULT false,           -- nothing survived verification
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,         -- MindmapArtifact.to_dict()
  pipeline_version text NOT NULL DEFAULT 'graph_pipeline_v0',
  verifier_versions jsonb NOT NULL DEFAULT '{}'::jsonb, -- entailment/calibrator/rules/evidence
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mindmap_graphs_user_entry
  ON public.mindmap_graphs (user_id, entry_date DESC NULLS LAST, updated_at DESC);

-- Idempotent batch upsert: one artifact per (user, source row, pipeline version).
CREATE UNIQUE INDEX IF NOT EXISTS uq_mindmap_graphs_source
  ON public.mindmap_graphs (user_id, source_table, source_id, pipeline_version);

ALTER TABLE public.mindmap_graphs ENABLE ROW LEVEL SECURITY;

-- Patient reads their own graphs. No provider policy (see header).
DO $$ BEGIN
  CREATE POLICY mindmap_graphs_select_own ON public.mindmap_graphs
    FOR SELECT USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMIT;
