-- ============================================================
-- Migration 011d: enable pg_cron and pg_net extensions
-- ============================================================
-- Backfill of a migration applied to production on 2026-05-13 (Supabase
-- version 20260513225105, name `enable_pg_cron_and_pg_net`) but never
-- committed. Recovered verbatim on 2026-08-06.
--
-- pg_cron runs the daily jobs (report generation etc.); pg_net lets those
-- jobs make outbound HTTP requests (e.g. calling the Anthropic API from
-- inside a scheduled function).
--
-- `create extension if not exists` is idempotent.
-- ============================================================

create extension if not exists pg_cron;
create extension if not exists pg_net;
