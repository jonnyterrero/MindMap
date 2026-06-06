-- ============================================================
-- Migration 018: Restrict lookup_provider to authenticated users.
-- The SECURITY DEFINER function from 017 was anon-executable; only
-- signed-in users need it (patient validating a provider code).
-- ============================================================

BEGIN;

REVOKE EXECUTE ON FUNCTION public.lookup_provider(uuid) FROM anon;
REVOKE EXECUTE ON FUNCTION public.lookup_provider(uuid) FROM public;
GRANT EXECUTE ON FUNCTION public.lookup_provider(uuid) TO authenticated;

COMMIT;
