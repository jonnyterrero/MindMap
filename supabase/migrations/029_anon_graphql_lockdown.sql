-- ============================================================
-- Migration 029: Lock anon out of the GraphQL/REST schema
-- ============================================================
-- Closes Supabase security advisor lint 0026
-- (pg_graphql_anon_table_exposed). Every base table in `public` is
-- currently SELECT-able by the `anon` role, so anyone holding the public
-- anon key can introspect the full GraphQL schema — table names, column
-- names, relationships — before signing in. Row Level Security still
-- blocks the *rows*, but the shape of the data model should not be
-- discoverable to an unauthenticated caller.
--
-- Observed state on prod (2026-10-08): `anon` holds the full table ACL
-- (arwdDxtm) on all 46 public tables, re-granted to future tables by the
-- default privileges owned by `postgres` and `supabase_admin`.
--
-- Fix:
--   1. Revoke ALL table privileges from `anon` across `public`.
--   2. Re-grant SELECT on `legal_documents` only — the one table that must
--      be publicly readable before sign-in (ToS / privacy / disclaimer).
--      RLS on that table already restricts rows to is_active = true
--      (migration 011).
--   3. Revoke the default privileges that would otherwise re-expose every
--      future table created by our own migrations (owner = postgres), and
--      best-effort the same for supabase_admin.
--
-- Deliberately NOT touched:
--   - `authenticated`: the app reads these tables as the signed-in user
--     via RLS (supabase-js / PostgREST). Revoking SELECT from
--     `authenticated` (lint 0027) would break the product and is a
--     separate decision — see docs/handoffs/2026-10-08-advisor-hardening.md.
--   - `anon` USAGE on schema `public`: kept, so the surviving
--     `legal_documents` grant remains reachable.
--   - sequences / functions: anon function EXECUTE was already tightened in
--     011b / 023; this migration is tables-only to match lint 0026.
--
-- After this runs, lint 0026 drops from 46 findings to 1 (legal_documents,
-- intentional). Idempotent: REVOKE/GRANT of an already-(un)set privilege is
-- a no-op, safe to re-run.
-- ============================================================

BEGIN;

-- 1) Strip every table/view privilege from anon in public.
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM anon;

-- 2) Restore the single intentional public read.
GRANT SELECT ON public.legal_documents TO anon;

-- 3) Stop future tables from re-granting anon. Default privileges are keyed
--    to the role that creates the object. Our migrations run as `postgres`;
--    Supabase internals may create objects as `supabase_admin`.
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
  REVOKE ALL ON TABLES FROM anon;

-- `postgres` is usually not permitted to alter supabase_admin's default
-- privileges on a managed instance; attempt it but do not fail the
-- migration if the grant is denied.
DO $$
BEGIN
  EXECUTE 'ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public REVOKE ALL ON TABLES FROM anon';
EXCEPTION
  WHEN insufficient_privilege THEN
    RAISE NOTICE 'Skipped supabase_admin default-privilege revoke (insufficient privilege). New tables created by supabase_admin, if any, may still grant anon; app tables are created by postgres and are covered above.';
  WHEN others THEN
    RAISE NOTICE 'Skipped supabase_admin default-privilege revoke (%).', SQLERRM;
END $$;

COMMIT;

-- ============================================================
-- Rollback (restores the pre-029, fully-exposed state — not recommended):
-- ============================================================
-- BEGIN;
--   GRANT ALL ON ALL TABLES IN SCHEMA public TO anon;
--   ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
--     GRANT ALL ON TABLES TO anon;
-- COMMIT;
