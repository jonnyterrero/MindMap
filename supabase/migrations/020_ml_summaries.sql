-- ============================================================
-- Migration 020: Tier-0 clinician summaries
-- The Python batch WRITES a structured, gated, evidence-cited summary
-- per user; the Next app only READS it. Additive & idempotent.
-- Writes are service-role only (no user/insert policy); users read their
-- own, providers read when granted read_reports (mirrors migration 017).
-- ============================================================

BEGIN;

CREATE TABLE IF NOT EXISTS public.mindmap_ml_summaries (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  period_start date,
  period_end date NOT NULL,                          -- as-of day the summary covers through
  abstained boolean NOT NULL DEFAULT false,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,         -- ClinicianSummary.to_dict()
  model_version text NOT NULL DEFAULT 'tier0_descriptive_v1',
  source text NOT NULL DEFAULT 'rules',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ml_summaries_user_time
  ON public.mindmap_ml_summaries (user_id, period_end DESC);

-- Idempotent batch upsert: one summary per (user, as-of day, model version).
CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_summaries_period
  ON public.mindmap_ml_summaries (user_id, period_end, model_version);

ALTER TABLE public.mindmap_ml_summaries ENABLE ROW LEVEL SECURITY;

-- Patient reads their own summaries.
DO $$ BEGIN
  CREATE POLICY ml_summaries_select_own ON public.mindmap_ml_summaries
    FOR SELECT USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Provider reads when granted read_reports (a summary is report-like).
DO $$ BEGIN
  CREATE POLICY ml_summaries_provider_read ON public.mindmap_ml_summaries
    FOR SELECT USING (
      EXISTS (
        SELECT 1 FROM public.mindmap_provider_access pa
        WHERE pa.provider_user_id = auth.uid()
          AND pa.patient_user_id = mindmap_ml_summaries.user_id
          AND pa.revoked_at IS NULL
          AND (pa.permissions->>'read_reports')::boolean IS TRUE
      )
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMIT;
