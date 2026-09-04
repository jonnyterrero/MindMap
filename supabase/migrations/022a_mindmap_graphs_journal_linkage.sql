-- ============================================================
-- Migration 022a: verified mindmap graphs — journal linkage + access fix.
-- ============================================================
-- Backfill of a migration applied to production on 2026-07-16 (Supabase
-- version 20260716214142, name `mindmap_graphs_journal_linkage`) but
-- never committed. Recovered verbatim on 2026-08-06.
--
-- A prior sketch of this table existed (empty) keyed on doc_id (a content
-- hash) with a provider read policy. Adapted additively:
--   * link rows to their source journal entry (source_table/source_id),
--     so edits refresh one row instead of accumulating per content hash,
--     and identical texts don't collide;
--   * SECURITY: drop provider read — payloads quote journal text verbatim
--     and mindmap_journal_entries is user-own only (017 has no journal
--     key); derived text must not be readable more broadly than its source.
-- Writes remain service-role only (no user insert/update policy).
--
-- All idempotent (ADD COLUMN / CREATE INDEX / DROP … IF EXISTS).
-- ============================================================

BEGIN;

ALTER TABLE public.mindmap_graphs
  ADD COLUMN IF NOT EXISTS source_table text NOT NULL DEFAULT 'mindmap_journal_entries',
  ADD COLUMN IF NOT EXISTS source_id uuid NOT NULL,
  ADD COLUMN IF NOT EXISTS entry_date date,
  ADD COLUMN IF NOT EXISTS content_sha text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS verifier_versions jsonb NOT NULL DEFAULT '{}'::jsonb;

-- One graph per source row per pipeline version (upsert conflict target).
CREATE UNIQUE INDEX IF NOT EXISTS uq_mindmap_graphs_source
  ON public.mindmap_graphs (user_id, source_table, source_id, pipeline_version);

-- The content-hash key is wrong for journal-linked rows: identical texts
-- collide and edited entries accumulate stale rows.
DROP INDEX IF EXISTS public.uq_mindmap_graphs_doc;

CREATE INDEX IF NOT EXISTS idx_mindmap_graphs_user_entry
  ON public.mindmap_graphs (user_id, entry_date DESC NULLS LAST, updated_at DESC);

-- SECURITY: journal-derived text is user-own only.
DROP POLICY IF EXISTS mindmap_graphs_provider_read ON public.mindmap_graphs;

COMMIT;
