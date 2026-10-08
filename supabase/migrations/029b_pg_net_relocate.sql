-- ============================================================
-- Migration 029b: Move pg_net out of the public schema
-- ============================================================
-- Closes Supabase security advisor lint 0014 (extension_in_public):
-- `pg_net` is registered in `public`. Advisors want extensions out of
-- `public` so their objects don't widen the app's public surface.
--
-- ⚠️  APPLY IN A MAINTENANCE WINDOW — NOT a routine grant change.
--
-- Why this is its own migration and not folded into 029:
--   pg_net is NON-RELOCATABLE (pg_available_extension_versions.relocatable
--   = false) and its control file pins no schema, so
--   `ALTER EXTENSION pg_net SET SCHEMA extensions` is REJECTED by Postgres.
--   The superuser workaround (temporarily flipping pg_extension.extrelocatable)
--   needs privileges `postgres` does not have on a managed instance.
--   The only migration-safe path is DROP + CREATE WITH SCHEMA extensions.
--
-- What DROP + CREATE does:
--   - pg_net's callable functions live in the dedicated `net` schema
--     (net.http_get / net.http_post / net.http_delete) and STAY there after
--     recreate — only the extension's *registration* schema moves to
--     `extensions`. So `net.http_post(...)` call sites (the pg_cron report /
--     ML-graph jobs) keep working by name.
--   - It drops and recreates the `net.*` queue/response tables. Those are
--     UNLOGGED and ephemeral (6-hour TTL); losing an in-flight request is
--     the risk — hence the window. Run when no cron job is firing.
--   - No CASCADE: if some object hard-depends on pg_net the DROP aborts the
--     whole transaction and names the dependent, rather than silently
--     dropping app objects. pg_cron jobs store their SQL as text (not a
--     catalog dependency) and plpgsql bodies calling net.* are not parsed
--     for deps, so a clean DROP is expected here.
--
-- LOWER-RISK ALTERNATIVE (if you'd rather not DROP/CREATE from SQL):
--   Supabase Dashboard → Database → Extensions → pg_net → disable, then
--   re-enable choosing schema `extensions`. Same end state, Supabase-managed.
--   If neither is acceptable, Supabase Support can run the extrelocatable
--   flip in place.
--
-- Idempotent: re-running after a successful move is a no-op (the guard sees
-- pg_net already in `extensions` and does nothing).
-- ============================================================

BEGIN;

DO $$
DECLARE
  cur_schema text;
BEGIN
  SELECT n.nspname
    INTO cur_schema
  FROM pg_extension e
  JOIN pg_namespace n ON n.oid = e.extnamespace
  WHERE e.extname = 'pg_net';

  IF cur_schema IS NULL THEN
    RAISE NOTICE 'pg_net is not installed; nothing to do.';
  ELSIF cur_schema = 'extensions' THEN
    RAISE NOTICE 'pg_net already registered in `extensions`; nothing to do.';
  ELSE
    RAISE NOTICE 'Relocating pg_net from `%` to `extensions` via DROP + CREATE...', cur_schema;
    EXECUTE 'DROP EXTENSION pg_net';
    EXECUTE 'CREATE EXTENSION pg_net WITH SCHEMA extensions';
    RAISE NOTICE 'pg_net relocated. Verify net.http_post is reachable and the next cron run succeeds.';
  END IF;
END $$;

COMMIT;

-- ============================================================
-- Rollback (moves pg_net back into public — only to undo this change):
-- ============================================================
-- BEGIN;
--   DROP EXTENSION IF EXISTS pg_net;
--   CREATE EXTENSION pg_net WITH SCHEMA public;
-- COMMIT;
