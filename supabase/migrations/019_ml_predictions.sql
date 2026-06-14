-- ============================================================
-- Migration 019: ML layer predictions
-- Extends public.mindmap_predictions (from 015) with the columns the
-- Python ML batch job writes. Additive & idempotent (IF NOT EXISTS).
-- The ML batch only WRITES here; the Next app only READS (existing
-- RLS + provider read_predictions gating from 015/017 still apply).
-- ============================================================

BEGIN;

ALTER TABLE public.mindmap_predictions
  ADD COLUMN IF NOT EXISTS uncertainty numeric(4,3),                       -- ~ 1 - confidence
  ADD COLUMN IF NOT EXISTS abstained boolean NOT NULL DEFAULT false,       -- model declined to estimate
  ADD COLUMN IF NOT EXISTS abstain_reason text,                            -- insufficient_history | low_confidence | ...
  ADD COLUMN IF NOT EXISTS evidence_citations jsonb NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS entry_date date,                               -- the as-of day the prediction was made for
  ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'ml';             -- rules | ml | hybrid

-- Provenance integrity: every row says how it was produced.
DO $$ BEGIN
  ALTER TABLE public.mindmap_predictions
    ADD CONSTRAINT chk_predictions_source CHECK (source IN ('rules','ml','hybrid'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Idempotent batch upsert key: one row per (user, type, as-of day, model version).
-- entry_date is NULL on legacy rows (015); NULLs are distinct, so those are
-- untouched while batch rows (entry_date set) get uniqueness for ON CONFLICT.
CREATE UNIQUE INDEX IF NOT EXISTS uq_predictions_batch
  ON public.mindmap_predictions (user_id, prediction_type, entry_date, model_version);

COMMIT;
