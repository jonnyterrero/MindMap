-- ============================================================
-- Migration 011b: Tighten SECURITY DEFINER function grants
-- ============================================================
-- Backfill of a migration applied to production on 2026-05-13 (Supabase
-- version 20260513210736, name `tighten_function_grants`) but never
-- committed. Recovered verbatim on 2026-08-06 via schema_migrations.
--
-- Removes EXECUTE from anon/authenticated on internal helpers that should
-- only ever be called from inside other SECURITY DEFINER functions, and
-- from anon on the three provider RPCs (which need auth.uid() to identify
-- the caller).
--
-- Lessons: migration 023 later re-CREATEd rpc_provider_get_entries and
-- rpc_provider_get_insights, which silently discarded this hardening --
-- Supabase's default privileges re-GRANT EXECUTE to anon on every new
-- function in `public`, and `REVOKE ALL FROM public` does NOT strip that
-- explicit role grant. 023 was patched (rls_policy_fixes_revoke_anon in
-- prod, inline in the repo) to re-revoke; any future DROP+CREATE on
-- these functions must revoke anon again in the same migration.
--
-- REVOKE is idempotent -- re-running against prod is a no-op.
-- ============================================================

REVOKE EXECUTE ON FUNCTION public.fn_log_provider_access(uuid, uuid, text, integer) FROM anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.fn_provider_can_read(uuid, uuid, text, date, date) FROM anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.fn_patient_provider_client_update_guard() FROM anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.handle_new_user() FROM anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.rpc_provider_get_entries(uuid, date, date) FROM anon;
REVOKE EXECUTE ON FUNCTION public.rpc_provider_get_insights(uuid, date, date) FROM anon;
REVOKE EXECUTE ON FUNCTION public.rpc_provider_get_journal_metadata(uuid, date, date) FROM anon;
