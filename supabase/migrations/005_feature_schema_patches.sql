-- Migration 005: Schema patches for new features
-- (Medications, Journal, Goals, Therapy, Insights, Consent, Body Sensations, Data Export)
-- Run in Supabase SQL Editor
-- https://supabase.com/dashboard/project/zunpccwjghwpiljwwjpv/sql

BEGIN;

-- =========================================================================
-- 1) BODY SENSATIONS — add missing `sensation` column
--    The app tracks body_part + sensation type (Pain, Tension, etc.)
-- =========================================================================

ALTER TABLE public.mindmap_body_sensations
  ADD COLUMN IF NOT EXISTS sensation text NOT NULL DEFAULT 'Pain'
    CHECK (sensation = ANY (ARRAY[
      'Pain', 'Tension', 'Numbness', 'Tingling', 'Burning',
      'Pressure', 'Throbbing', 'Aching', 'Stiffness', 'Heaviness',
      'Lightness', 'Warmth', 'Coldness', 'Nausea'
    ]));

-- =========================================================================
-- 2) INSIGHTS — add `recommendation` column + unique constraint for upsert
--    The insights engine stores an actionable recommendation per insight.
--    Also widen the risk_level CHECK to include 'stable', 'concerning', 'unknown'
--    which the mood_trend insight uses.
-- =========================================================================

ALTER TABLE public.mindmap_insights
  ADD COLUMN IF NOT EXISTS recommendation text;

-- Widen risk_level to include new values from the mood_trend insight type
ALTER TABLE public.mindmap_insights
  DROP CONSTRAINT IF EXISTS mindmap_insights_risk_level_check;

ALTER TABLE public.mindmap_insights
  ADD CONSTRAINT mindmap_insights_risk_level_check
    CHECK (risk_level = ANY (ARRAY[
      'low', 'moderate', 'high', 'critical',
      'stable', 'concerning', 'unknown'
    ]));

-- Unique constraint for upsert: one insight per type per entry per user
ALTER TABLE public.mindmap_insights
  DROP CONSTRAINT IF EXISTS mindmap_insights_user_entry_type_uq;

ALTER TABLE public.mindmap_insights
  ADD CONSTRAINT mindmap_insights_user_entry_type_uq
    UNIQUE (user_id, entry_id, insight_type);

-- =========================================================================
-- 3) CONSENT RECORDS — rename `granted` → `consent_given`
--    The app code uses `consent_given` everywhere.
--    DDL renames are not blocked by the immutability trigger.
-- =========================================================================

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'consent_records'
      AND column_name = 'granted'
  ) THEN
    ALTER TABLE public.consent_records RENAME COLUMN granted TO consent_given;
  END IF;
END;
$$;

-- =========================================================================
-- 4) MEDICATION ADHERENCE — unique constraint for upsert
--    One adherence record per medication per entry (per day).
-- =========================================================================

ALTER TABLE public.mindmap_medication_adherence
  DROP CONSTRAINT IF EXISTS mindmap_med_adherence_schedule_entry_uq;

ALTER TABLE public.mindmap_medication_adherence
  ADD CONSTRAINT mindmap_med_adherence_schedule_entry_uq
    UNIQUE (medication_schedule_id, entry_id);

-- =========================================================================
-- 5) INDEXES for new query patterns
-- =========================================================================

-- Insights: fast lookup for latest per user
CREATE INDEX IF NOT EXISTS idx_insights_user_computed
  ON public.mindmap_insights (user_id, computed_at DESC);

-- Therapy sessions: user lookup
CREATE INDEX IF NOT EXISTS idx_therapy_sessions_user
  ON public.mindmap_therapy_sessions (user_id, session_date DESC);

-- Goals: active goals per user
CREATE INDEX IF NOT EXISTS idx_goals_user_completed
  ON public.mindmap_goals (user_id, is_completed, created_at DESC);

-- Consent: fast check for "has this user consented?"
CREATE INDEX IF NOT EXISTS idx_consent_user_given
  ON public.consent_records (user_id, consent_type, consent_given);

-- Data deletion: pending requests
CREATE INDEX IF NOT EXISTS idx_deletion_pending
  ON public.data_deletion_requests (status, created_at DESC);

-- Medication adherence: by entry
CREATE INDEX IF NOT EXISTS idx_med_adherence_entry
  ON public.mindmap_medication_adherence (entry_id);

-- =========================================================================
-- 6) RLS policies for tables that might be missing coverage
-- =========================================================================

-- Therapy sessions (ensure exists)
ALTER TABLE public.mindmap_therapy_sessions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own therapy sessions" ON public.mindmap_therapy_sessions;
CREATE POLICY "Users manage own therapy sessions"
  ON public.mindmap_therapy_sessions FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Goals
ALTER TABLE public.mindmap_goals ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own goals" ON public.mindmap_goals;
CREATE POLICY "Users manage own goals"
  ON public.mindmap_goals FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Data exports
ALTER TABLE public.mindmap_data_exports ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own exports" ON public.mindmap_data_exports;
CREATE POLICY "Users manage own exports"
  ON public.mindmap_data_exports FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

COMMIT;
