-- Migration 003: RLS on all core tables + indexes + profile auto-creation
-- Run in Supabase SQL Editor
-- https://supabase.com/dashboard/project/zunpccwjghwpiljwwjpv/sql

BEGIN;

-- =========================================================================
-- 1) ROW LEVEL SECURITY — core tables
--    Without this, any authenticated user can see everyone's data.
-- =========================================================================

-- profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own profile" ON public.profiles;
CREATE POLICY "Users manage own profile"
  ON public.profiles FOR ALL
  USING (auth.uid() = id)
  WITH CHECK (auth.uid() = id);

-- mindmap_entries
ALTER TABLE public.mindmap_entries ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own entries" ON public.mindmap_entries;
CREATE POLICY "Users manage own entries"
  ON public.mindmap_entries FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- mindmap_body_sensations (via entry ownership)
ALTER TABLE public.mindmap_body_sensations ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own body sensations" ON public.mindmap_body_sensations;
CREATE POLICY "Users manage own body sensations"
  ON public.mindmap_body_sensations FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM public.mindmap_entries e
      WHERE e.id = mindmap_body_sensations.entry_id
        AND e.user_id = auth.uid()
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.mindmap_entries e
      WHERE e.id = mindmap_body_sensations.entry_id
        AND e.user_id = auth.uid()
    )
  );

-- mindmap_meds (via entry ownership)
ALTER TABLE public.mindmap_meds ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own meds" ON public.mindmap_meds;
CREATE POLICY "Users manage own meds"
  ON public.mindmap_meds FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM public.mindmap_entries e
      WHERE e.id = mindmap_meds.entry_id
        AND e.user_id = auth.uid()
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.mindmap_entries e
      WHERE e.id = mindmap_meds.entry_id
        AND e.user_id = auth.uid()
    )
  );

-- mindmap_medication_schedule
ALTER TABLE public.mindmap_medication_schedule ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own med schedule" ON public.mindmap_medication_schedule;
CREATE POLICY "Users manage own med schedule"
  ON public.mindmap_medication_schedule FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- mindmap_medication_adherence (via schedule ownership)
ALTER TABLE public.mindmap_medication_adherence ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own med adherence" ON public.mindmap_medication_adherence;
CREATE POLICY "Users manage own med adherence"
  ON public.mindmap_medication_adherence FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM public.mindmap_medication_schedule ms
      WHERE ms.id = mindmap_medication_adherence.medication_schedule_id
        AND ms.user_id = auth.uid()
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.mindmap_medication_schedule ms
      WHERE ms.id = mindmap_medication_adherence.medication_schedule_id
        AND ms.user_id = auth.uid()
    )
  );

-- mindmap_routines
ALTER TABLE public.mindmap_routines ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own routines" ON public.mindmap_routines;
CREATE POLICY "Users manage own routines"
  ON public.mindmap_routines FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- mindmap_entry_routines (via entry ownership)
ALTER TABLE public.mindmap_entry_routines ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own entry routines" ON public.mindmap_entry_routines;
CREATE POLICY "Users manage own entry routines"
  ON public.mindmap_entry_routines FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM public.mindmap_entries e
      WHERE e.id = mindmap_entry_routines.entry_id
        AND e.user_id = auth.uid()
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.mindmap_entries e
      WHERE e.id = mindmap_entry_routines.entry_id
        AND e.user_id = auth.uid()
    )
  );

-- mindmap_journal_entries
ALTER TABLE public.mindmap_journal_entries ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own journal" ON public.mindmap_journal_entries;
CREATE POLICY "Users manage own journal"
  ON public.mindmap_journal_entries FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- mindmap_therapy_sessions
ALTER TABLE public.mindmap_therapy_sessions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own therapy sessions" ON public.mindmap_therapy_sessions;
CREATE POLICY "Users manage own therapy sessions"
  ON public.mindmap_therapy_sessions FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- mindmap_goals
ALTER TABLE public.mindmap_goals ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own goals" ON public.mindmap_goals;
CREATE POLICY "Users manage own goals"
  ON public.mindmap_goals FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- mindmap_triggers
ALTER TABLE public.mindmap_triggers ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own triggers" ON public.mindmap_triggers;
CREATE POLICY "Users manage own triggers"
  ON public.mindmap_triggers FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- mindmap_trigger_events (via trigger ownership)
ALTER TABLE public.mindmap_trigger_events ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own trigger events" ON public.mindmap_trigger_events;
CREATE POLICY "Users manage own trigger events"
  ON public.mindmap_trigger_events FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM public.mindmap_triggers t
      WHERE t.id = mindmap_trigger_events.trigger_id
        AND t.user_id = auth.uid()
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.mindmap_triggers t
      WHERE t.id = mindmap_trigger_events.trigger_id
        AND t.user_id = auth.uid()
    )
  );

-- mindmap_reminders
ALTER TABLE public.mindmap_reminders ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own reminders" ON public.mindmap_reminders;
CREATE POLICY "Users manage own reminders"
  ON public.mindmap_reminders FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- mindmap_data_exports
ALTER TABLE public.mindmap_data_exports ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users manage own exports" ON public.mindmap_data_exports;
CREATE POLICY "Users manage own exports"
  ON public.mindmap_data_exports FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- =========================================================================
-- 2) UNIQUE CONSTRAINT — one entry per user per day
-- =========================================================================

ALTER TABLE public.mindmap_entries
  DROP CONSTRAINT IF EXISTS mindmap_entries_user_date_uq;
ALTER TABLE public.mindmap_entries
  ADD CONSTRAINT mindmap_entries_user_date_uq UNIQUE (user_id, entry_date);

-- =========================================================================
-- 3) AUTO-CREATE PROFILE on signup
-- =========================================================================

CREATE OR REPLACE FUNCTION public.handle_new_user()
  RETURNS trigger AS $$
BEGIN
  INSERT INTO public.profiles (id, display_name)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data ->> 'display_name', split_part(NEW.email, '@', 1))
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- =========================================================================
-- 4) INDEXES for common query patterns
-- =========================================================================

-- Daily entry lookups (Today page)
CREATE INDEX IF NOT EXISTS idx_entries_user_date
  ON public.mindmap_entries (user_id, entry_date DESC);

-- Entry children (body sensations, meds)
CREATE INDEX IF NOT EXISTS idx_body_sensations_entry
  ON public.mindmap_body_sensations (entry_id);

CREATE INDEX IF NOT EXISTS idx_meds_entry
  ON public.mindmap_meds (entry_id);

-- Routines per user
CREATE INDEX IF NOT EXISTS idx_routines_user_active
  ON public.mindmap_routines (user_id, is_active);

-- Entry routines lookup
CREATE INDEX IF NOT EXISTS idx_entry_routines_entry
  ON public.mindmap_entry_routines (entry_id);

-- Medication schedule per user
CREATE INDEX IF NOT EXISTS idx_med_schedule_user
  ON public.mindmap_medication_schedule (user_id, is_active);

-- Medication adherence per schedule
CREATE INDEX IF NOT EXISTS idx_med_adherence_schedule
  ON public.mindmap_medication_adherence (medication_schedule_id);

-- Journal entries per user
CREATE INDEX IF NOT EXISTS idx_journal_user_date
  ON public.mindmap_journal_entries (user_id, entry_date DESC);

-- Therapy sessions per user
CREATE INDEX IF NOT EXISTS idx_therapy_user_date
  ON public.mindmap_therapy_sessions (user_id, session_date DESC);

-- Goals per user
CREATE INDEX IF NOT EXISTS idx_goals_user_active
  ON public.mindmap_goals (user_id, is_active);

-- Triggers per user
CREATE INDEX IF NOT EXISTS idx_triggers_user_active
  ON public.mindmap_triggers (user_id, is_active);

-- Trigger events per trigger
CREATE INDEX IF NOT EXISTS idx_trigger_events_trigger
  ON public.mindmap_trigger_events (trigger_id, triggered_at DESC);

-- Reminders per user
CREATE INDEX IF NOT EXISTS idx_reminders_user_active
  ON public.mindmap_reminders (user_id, is_active);

-- Exports per user
CREATE INDEX IF NOT EXISTS idx_exports_user
  ON public.mindmap_data_exports (user_id, created_at DESC);

-- Sharing: provider clients by patient
CREATE INDEX IF NOT EXISTS idx_provider_clients_patient
  ON public.provider_clients (patient_user_id, status);

-- Sharing: data shares by patient
CREATE INDEX IF NOT EXISTS idx_data_shares_patient
  ON public.data_shares (patient_user_id, is_active);

-- Audit log by patient
CREATE INDEX IF NOT EXISTS idx_audit_log_patient
  ON public.sharing_audit_log (patient_user_id, created_at DESC);

-- =========================================================================
-- 5) updated_at auto-refresh trigger
-- =========================================================================

CREATE OR REPLACE FUNCTION public.set_updated_at()
  RETURNS trigger AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to all tables with updated_at
DO $$
DECLARE
  tbl text;
BEGIN
  FOR tbl IN
    SELECT unnest(ARRAY[
      'mindmap_entries',
      'mindmap_medication_schedule',
      'mindmap_journal_entries',
      'mindmap_therapy_sessions',
      'mindmap_goals',
      'mindmap_triggers',
      'mindmap_reminders',
      'mindmap_data_exports',
      'data_deletion_requests',
      'provider_orgs',
      'provider_profiles',
      'provider_clients',
      'data_shares'
    ])
  LOOP
    EXECUTE format('DROP TRIGGER IF EXISTS set_updated_at ON public.%I', tbl);
    EXECUTE format(
      'CREATE TRIGGER set_updated_at BEFORE UPDATE ON public.%I '
      'FOR EACH ROW EXECUTE FUNCTION public.set_updated_at()',
      tbl
    );
  END LOOP;
END;
$$;

COMMIT;
