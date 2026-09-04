-- ============================================================
-- Migration 011a: DB cleanup
-- ============================================================
-- Backfill of a migration that was applied directly to production on
-- 2026-05-13 (Supabase version 20260513210020, name
-- `cleanup_views_and_functions`) but never landed in the repo. Recovered
-- verbatim on 2026-08-06 via schema_migrations so `main` reproduces prod.
--
-- Drops the v_shared_* provider views from migration 004, retires an old
-- set_updated_at() trigger function in favor of fn_set_updated_at(), and
-- pins `search_path = public, pg_temp` on the seven functions listed
-- below so a SET search_path attack cannot resolve their references.
--
-- Every statement is idempotent (DROP … IF EXISTS, ALTER FUNCTION SET);
-- re-running against prod is a no-op.
-- ============================================================

DROP VIEW IF EXISTS public.v_shared_entries_full CASCADE;
DROP VIEW IF EXISTS public.v_shared_entries_no_notes CASCADE;
DROP VIEW IF EXISTS public.v_shared_entries_summary CASCADE;
DROP VIEW IF EXISTS public.v_shared_journal_full CASCADE;
DROP VIEW IF EXISTS public.v_shared_journal_metadata CASCADE;

DROP TRIGGER IF EXISTS set_updated_at ON public.mindmap_entries;
DROP TRIGGER IF EXISTS set_updated_at ON public.mindmap_medication_schedule;
DROP TRIGGER IF EXISTS set_updated_at ON public.mindmap_journal_entries;
DROP TRIGGER IF EXISTS set_updated_at ON public.mindmap_therapy_sessions;
DROP TRIGGER IF EXISTS set_updated_at ON public.mindmap_goals;
DROP TRIGGER IF EXISTS set_updated_at ON public.mindmap_triggers;
DROP TRIGGER IF EXISTS set_updated_at ON public.mindmap_reminders;
DROP TRIGGER IF EXISTS set_updated_at ON public.data_deletion_requests;
DROP TRIGGER IF EXISTS set_updated_at ON public.provider_orgs;
DROP TRIGGER IF EXISTS set_updated_at ON public.provider_profiles;
DROP TRIGGER IF EXISTS set_updated_at ON public.provider_clients;
DROP TRIGGER IF EXISTS set_updated_at ON public.data_shares;

DROP TRIGGER IF EXISTS trg_set_updated_at_entries ON public.mindmap_entries;
DROP TRIGGER IF EXISTS trg_set_updated_at_medication_schedule ON public.mindmap_medication_schedule;
DROP TRIGGER IF EXISTS trg_set_updated_at_goals ON public.mindmap_goals;
DROP TRIGGER IF EXISTS trg_set_updated_at_triggers ON public.mindmap_triggers;
DROP TRIGGER IF EXISTS trg_set_updated_at_therapy_sessions ON public.mindmap_therapy_sessions;
DROP TRIGGER IF EXISTS trg_set_updated_at_journal_entries ON public.mindmap_journal_entries;
DROP TRIGGER IF EXISTS trg_set_updated_at_reminders ON public.mindmap_reminders;

DROP TRIGGER IF EXISTS set_updated_at ON public.mindmap_data_exports;
DROP TRIGGER IF EXISTS trg_updated_at_mindmap_data_exports ON public.mindmap_data_exports;
CREATE TRIGGER trg_updated_at_mindmap_data_exports
  BEFORE UPDATE ON public.mindmap_data_exports
  FOR EACH ROW EXECUTE FUNCTION public.fn_set_updated_at();

DROP FUNCTION IF EXISTS public.set_updated_at();

ALTER FUNCTION public.fn_bump_sync_version() SET search_path = public, pg_temp;
ALTER FUNCTION public.fn_patient_provider_client_update_guard() SET search_path = public, pg_temp;
ALTER FUNCTION public.fn_security_audit_immutable() SET search_path = public, pg_temp;
ALTER FUNCTION public.fn_set_updated_at() SET search_path = public, pg_temp;
ALTER FUNCTION public.handle_new_user() SET search_path = public, pg_temp;
ALTER FUNCTION public.prevent_audit_mutation() SET search_path = public, pg_temp;
ALTER FUNCTION public.prevent_consent_mutation() SET search_path = public, pg_temp;
