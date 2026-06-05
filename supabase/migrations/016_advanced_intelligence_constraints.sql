-- ============================================================
-- Migration 016: Advanced-intelligence schema hardening
-- Data-integrity CHECK constraints on enum-like text columns +
-- index refinements. Safe: tables from 015 are new and empty.
-- (database-design pass on migration 015.)
-- ============================================================

BEGIN;

-- Enum integrity (the 015 columns were free text with only comments).
DO $$ BEGIN
  ALTER TABLE public.mindmap_predictions
    ADD CONSTRAINT chk_predictions_type
    CHECK (prediction_type IN ('migraine','anxiety','mood','pain_flare'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.mindmap_wearable_sources
    ADD CONSTRAINT chk_wearable_source_type
    CHECK (source_type IN ('apple_health','health_connect','fitbit','oura','garmin','whoop'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.mindmap_wearable_data
    ADD CONSTRAINT chk_wearable_metric_type
    CHECK (metric_type IN ('hrv','sleep_score','resting_hr','steps','spo2','temperature','calories','respiratory_rate'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  ALTER TABLE public.mindmap_crisis_events
    ADD CONSTRAINT chk_crisis_trigger_source
    CHECK (trigger_source IS NULL OR trigger_source IN ('ai_message','journal_entry','voice','manual'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Offline queue: drain pending items in creation order -> index should carry created_at.
CREATE INDEX IF NOT EXISTS idx_offline_queue_user_status_created
  ON public.mindmap_offline_queue (user_id, status, created_at);
DROP INDEX IF EXISTS public.idx_offline_queue_user_status;

-- FK index for cascade/SET NULL + source-scoped reads.
CREATE INDEX IF NOT EXISTS idx_wearable_data_source
  ON public.mindmap_wearable_data (source_id);

COMMIT;
