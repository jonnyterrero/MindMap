-- ============================================================
-- Migration 014: AI journal reflection (optional, opt-in)
-- Adds an opt-in flag + a place to persist AI-generated reflections
-- so they aren't recomputed (and re-billed) on every view.
--
-- Reflections are gentle, non-diagnostic summaries + a reflection
-- question + emotional-theme tags. Disabled by default.
-- ============================================================

BEGIN;

ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS ai_reflection_enabled boolean NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS public.mindmap_journal_ai_analysis (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  journal_entry_id uuid NOT NULL REFERENCES public.mindmap_journal_entries(id) ON DELETE CASCADE,
  summary text,
  reflection_question text,
  tags text[] NOT NULL DEFAULT '{}',
  model text,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (journal_entry_id)
);

CREATE INDEX IF NOT EXISTS idx_journal_ai_analysis_user
  ON public.mindmap_journal_ai_analysis (user_id);

ALTER TABLE public.mindmap_journal_ai_analysis ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  CREATE POLICY journal_ai_analysis_select_own ON public.mindmap_journal_ai_analysis
    FOR SELECT USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE POLICY journal_ai_analysis_insert_own ON public.mindmap_journal_ai_analysis
    FOR INSERT WITH CHECK (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE POLICY journal_ai_analysis_update_own ON public.mindmap_journal_ai_analysis
    FOR UPDATE USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE POLICY journal_ai_analysis_delete_own ON public.mindmap_journal_ai_analysis
    FOR DELETE USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMIT;
