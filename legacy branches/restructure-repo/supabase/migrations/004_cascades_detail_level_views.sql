-- Migration 004: Cascade deletes, detail_level on shares, audit FKs, provider views
-- Run in Supabase SQL Editor
-- https://supabase.com/dashboard/project/zunpccwjghwpiljwwjpv/sql

BEGIN;

-- =========================================================================
-- 1) CASCADE DELETES on child tables off mindmap_entries
--    When an entry is deleted (or via deletion request), children go too.
-- =========================================================================

ALTER TABLE public.mindmap_body_sensations
  DROP CONSTRAINT mindmap_body_sensations_entry_id_fkey,
  ADD CONSTRAINT mindmap_body_sensations_entry_id_fkey
    FOREIGN KEY (entry_id) REFERENCES public.mindmap_entries(id) ON DELETE CASCADE;

ALTER TABLE public.mindmap_meds
  DROP CONSTRAINT mindmap_meds_entry_id_fkey,
  ADD CONSTRAINT mindmap_meds_entry_id_fkey
    FOREIGN KEY (entry_id) REFERENCES public.mindmap_entries(id) ON DELETE CASCADE;

ALTER TABLE public.mindmap_entry_routines
  DROP CONSTRAINT mindmap_entry_routines_entry_id_fkey,
  ADD CONSTRAINT mindmap_entry_routines_entry_id_fkey
    FOREIGN KEY (entry_id) REFERENCES public.mindmap_entries(id) ON DELETE CASCADE;

ALTER TABLE public.mindmap_medication_adherence
  DROP CONSTRAINT mindmap_medication_adherence_entry_id_fkey,
  ADD CONSTRAINT mindmap_medication_adherence_entry_id_fkey
    FOREIGN KEY (entry_id) REFERENCES public.mindmap_entries(id) ON DELETE CASCADE;

ALTER TABLE public.mindmap_trigger_events
  DROP CONSTRAINT mindmap_trigger_events_entry_id_fkey,
  ADD CONSTRAINT mindmap_trigger_events_entry_id_fkey
    FOREIGN KEY (entry_id) REFERENCES public.mindmap_entries(id) ON DELETE CASCADE;

ALTER TABLE public.mindmap_insights
  DROP CONSTRAINT mindmap_insights_entry_id_fkey,
  ADD CONSTRAINT mindmap_insights_entry_id_fkey
    FOREIGN KEY (entry_id) REFERENCES public.mindmap_entries(id) ON DELETE CASCADE;

-- Also cascade medication_adherence off schedule deletion
ALTER TABLE public.mindmap_medication_adherence
  DROP CONSTRAINT mindmap_medication_adherence_medication_schedule_id_fkey,
  ADD CONSTRAINT mindmap_medication_adherence_medication_schedule_id_fkey
    FOREIGN KEY (medication_schedule_id) REFERENCES public.mindmap_medication_schedule(id) ON DELETE CASCADE;

-- Cascade trigger_events off trigger deletion
ALTER TABLE public.mindmap_trigger_events
  DROP CONSTRAINT mindmap_trigger_events_trigger_id_fkey,
  ADD CONSTRAINT mindmap_trigger_events_trigger_id_fkey
    FOREIGN KEY (trigger_id) REFERENCES public.mindmap_triggers(id) ON DELETE CASCADE;

-- =========================================================================
-- 2) DETAIL LEVEL on data_shares
--    Controls whether providers see notes/journal content.
--    summary  = aggregates/trends only
--    no_notes = structured fields, excludes notes/journal content
--    full     = everything including notes (explicit patient consent)
-- =========================================================================

ALTER TABLE public.data_shares
  ADD COLUMN IF NOT EXISTS detail_level text NOT NULL DEFAULT 'summary'
    CHECK (detail_level = ANY (ARRAY['summary', 'no_notes', 'full']));

COMMENT ON COLUMN public.data_shares.detail_level
  IS 'Controls note/journal visibility: summary=aggregates, no_notes=structured only, full=includes notes';

-- =========================================================================
-- 3) AUDIT LOG FK constraints
--    Links back to provider_clients and data_shares for integrity.
--    No cascade — audit records survive even if the relationship is deleted.
-- =========================================================================

ALTER TABLE public.sharing_audit_log
  ADD CONSTRAINT IF NOT EXISTS sharing_audit_log_provider_client_fkey
    FOREIGN KEY (provider_client_id) REFERENCES public.provider_clients(id);

ALTER TABLE public.sharing_audit_log
  ADD CONSTRAINT IF NOT EXISTS sharing_audit_log_data_share_fkey
    FOREIGN KEY (data_share_id) REFERENCES public.data_shares(id);

-- =========================================================================
-- 4) PROVIDER-SAFE VIEWS
--    Providers query these views, never the base tables directly.
--    Views respect data_shares scope, date range, and detail_level.
-- =========================================================================

-- Shared entries: structured data, no notes
CREATE OR REPLACE VIEW public.v_shared_entries_no_notes AS
SELECT
  e.user_id AS patient_user_id,
  e.entry_date,
  e.sleep_minutes,
  e.sleep_quality,
  e.bed_time,
  e.wake_time,
  e.hrv,
  e.mood_valence,
  e.anxiety,
  e.depression,
  e.mania,
  e.focus,
  e.productivity,
  e.therapy_minutes,
  e.outside_minutes,
  e.migraine,
  e.migraine_intensity,
  e.migraine_aura,
  ds.provider_client_id,
  ds.id AS data_share_id,
  pc.org_id
FROM public.data_shares ds
JOIN public.provider_clients pc ON pc.id = ds.provider_client_id
JOIN public.mindmap_entries e ON e.user_id = ds.patient_user_id
WHERE ds.is_active = true
  AND pc.status = 'active'
  AND ds.resource_type IN ('entries', 'all')
  AND ds.detail_level IN ('no_notes', 'full')
  AND (ds.date_range_start IS NULL OR e.entry_date >= ds.date_range_start)
  AND (ds.date_range_end IS NULL OR e.entry_date <= ds.date_range_end);

-- Shared entries: full detail including notes
CREATE OR REPLACE VIEW public.v_shared_entries_full AS
SELECT
  e.user_id AS patient_user_id,
  e.entry_date,
  e.sleep_minutes,
  e.sleep_quality,
  e.bed_time,
  e.wake_time,
  e.hrv,
  e.mood_valence,
  e.anxiety,
  e.depression,
  e.mania,
  e.focus,
  e.productivity,
  e.therapy_minutes,
  e.outside_minutes,
  e.migraine,
  e.migraine_intensity,
  e.migraine_aura,
  e.notes,
  ds.provider_client_id,
  ds.id AS data_share_id,
  pc.org_id
FROM public.data_shares ds
JOIN public.provider_clients pc ON pc.id = ds.provider_client_id
JOIN public.mindmap_entries e ON e.user_id = ds.patient_user_id
WHERE ds.is_active = true
  AND pc.status = 'active'
  AND ds.resource_type IN ('entries', 'all')
  AND ds.detail_level = 'full'
  AND (ds.date_range_start IS NULL OR e.entry_date >= ds.date_range_start)
  AND (ds.date_range_end IS NULL OR e.entry_date <= ds.date_range_end);

-- Shared journal: metadata only (title, date, mood tags — never content)
CREATE OR REPLACE VIEW public.v_shared_journal_metadata AS
SELECT
  j.user_id AS patient_user_id,
  j.entry_date,
  j.entry_time,
  j.title,
  j.mood_tags,
  ds.provider_client_id,
  ds.id AS data_share_id,
  pc.org_id
FROM public.data_shares ds
JOIN public.provider_clients pc ON pc.id = ds.provider_client_id
JOIN public.mindmap_journal_entries j ON j.user_id = ds.patient_user_id
WHERE ds.is_active = true
  AND pc.status = 'active'
  AND ds.resource_type IN ('journal', 'all')
  AND j.is_private = false
  AND (ds.date_range_start IS NULL OR j.entry_date >= ds.date_range_start)
  AND (ds.date_range_end IS NULL OR j.entry_date <= ds.date_range_end);

-- Shared journal: full content (only when detail_level = 'full' AND not private)
CREATE OR REPLACE VIEW public.v_shared_journal_full AS
SELECT
  j.user_id AS patient_user_id,
  j.entry_date,
  j.entry_time,
  j.title,
  j.content,
  j.mood_tags,
  ds.provider_client_id,
  ds.id AS data_share_id,
  pc.org_id
FROM public.data_shares ds
JOIN public.provider_clients pc ON pc.id = ds.provider_client_id
JOIN public.mindmap_journal_entries j ON j.user_id = ds.patient_user_id
WHERE ds.is_active = true
  AND pc.status = 'active'
  AND ds.resource_type IN ('journal', 'all')
  AND ds.detail_level = 'full'
  AND j.is_private = false
  AND (ds.date_range_start IS NULL OR j.entry_date >= ds.date_range_start)
  AND (ds.date_range_end IS NULL OR j.entry_date <= ds.date_range_end);

-- Shared insights: providers see explainable risk scores for their patients
CREATE OR REPLACE VIEW public.v_shared_insights AS
SELECT
  i.user_id AS patient_user_id,
  i.insight_type,
  i.risk_level,
  i.score,
  i.reasons,
  i.signals,
  i.summary,
  i.computed_at,
  ds.provider_client_id,
  ds.id AS data_share_id,
  pc.org_id
FROM public.data_shares ds
JOIN public.provider_clients pc ON pc.id = ds.provider_client_id
JOIN public.mindmap_insights i ON i.user_id = ds.patient_user_id
WHERE ds.is_active = true
  AND pc.status = 'active'
  AND ds.resource_type IN ('entries', 'all')
  AND (ds.date_range_start IS NULL OR i.computed_at::date >= ds.date_range_start)
  AND (ds.date_range_end IS NULL OR i.computed_at::date <= ds.date_range_end);

-- Shared medication schedule (no adherence detail, just what's prescribed)
CREATE OR REPLACE VIEW public.v_shared_medications AS
SELECT
  ms.user_id AS patient_user_id,
  ms.name,
  ms.dosage,
  ms.dose_mg,
  ms.frequency,
  ms.is_active,
  ms.start_date,
  ms.end_date,
  ds.provider_client_id,
  ds.id AS data_share_id,
  pc.org_id
FROM public.data_shares ds
JOIN public.provider_clients pc ON pc.id = ds.provider_client_id
JOIN public.mindmap_medication_schedule ms ON ms.user_id = ds.patient_user_id
WHERE ds.is_active = true
  AND pc.status = 'active'
  AND ds.resource_type IN ('medications', 'all');

COMMIT;
