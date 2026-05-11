-- ============================================================
-- Migration 006: Mobile sync columns + explicit array types
-- Adds deleted_at, client_updated_at, sync_version to user-owned
-- domain tables. Normalises ARRAY columns to explicit element types.
-- All operations are additive and idempotent.
-- ============================================================

BEGIN;

-- ============================================================
-- 1) MOBILE SYNC COLUMNS
--    Applied uniformly to every user-owned mutable domain table.
-- ============================================================

DO $$
DECLARE
  tbl TEXT;
BEGIN
  FOR tbl IN
    SELECT unnest(ARRAY[
      'mindmap_entries',
      'mindmap_journal_entries',
      'mindmap_goals',
      'mindmap_medication_schedule',
      'mindmap_medication_adherence',
      'mindmap_reminders',
      'mindmap_routines',
      'mindmap_therapy_sessions',
      'mindmap_triggers',
      'mindmap_body_sensations',
      'mindmap_meds'
    ])
  LOOP
    EXECUTE format($f$
      ALTER TABLE public.%I
        ADD COLUMN IF NOT EXISTS deleted_at timestamptz,
        ADD COLUMN IF NOT EXISTS client_updated_at timestamptz,
        ADD COLUMN IF NOT EXISTS sync_version bigint NOT NULL DEFAULT 1
    $f$, tbl);

    -- Index supports incremental sync queries: WHERE updated_at > :since
    EXECUTE format($f$
      CREATE INDEX IF NOT EXISTS %I ON public.%I (user_id, updated_at DESC)
      WHERE deleted_at IS NULL
    $f$, 'idx_' || tbl || '_user_updated_live', tbl);
  END LOOP;
END;
$$;

-- sync_version auto-increment trigger
CREATE OR REPLACE FUNCTION fn_bump_sync_version()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.sync_version = OLD.sync_version THEN
    NEW.sync_version := OLD.sync_version + 1;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
  tbl TEXT;
  trigger_name TEXT;
BEGIN
  FOR tbl IN
    SELECT unnest(ARRAY[
      'mindmap_entries',
      'mindmap_journal_entries',
      'mindmap_goals',
      'mindmap_medication_schedule',
      'mindmap_medication_adherence',
      'mindmap_reminders',
      'mindmap_routines',
      'mindmap_therapy_sessions',
      'mindmap_triggers',
      'mindmap_body_sensations',
      'mindmap_meds'
    ])
  LOOP
    trigger_name := 'trg_sync_version_' || tbl;
    EXECUTE format(
      'DROP TRIGGER IF EXISTS %I ON public.%I; '
      'CREATE TRIGGER %I BEFORE UPDATE ON public.%I '
      'FOR EACH ROW EXECUTE FUNCTION fn_bump_sync_version();',
      trigger_name, tbl, trigger_name, tbl
    );
  END LOOP;
END;
$$;


-- ============================================================
-- 2) EXPLICIT ARRAY TYPES
--    Postgres accepts ARRAY without element type only via inference.
--    Make them explicit where they were ambiguous in the legacy schema.
-- ============================================================

-- mood_tags: free-form text labels
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'mindmap_journal_entries'
      AND column_name = 'mood_tags'
      AND data_type = 'ARRAY'
      AND udt_name NOT IN ('_text')
  ) THEN
    ALTER TABLE public.mindmap_journal_entries
      ALTER COLUMN mood_tags TYPE text[] USING mood_tags::text[];
  END IF;
END;
$$;

ALTER TABLE public.mindmap_journal_entries
  ALTER COLUMN mood_tags SET DEFAULT '{}'::text[];

-- days_of_week: 1..7 ISO weekday numbers
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'mindmap_reminders'
      AND column_name = 'days_of_week'
      AND data_type = 'ARRAY'
      AND udt_name NOT IN ('_int2', '_int4')
  ) THEN
    ALTER TABLE public.mindmap_reminders
      ALTER COLUMN days_of_week TYPE smallint[] USING days_of_week::smallint[];
  END IF;
END;
$$;

DO $$ BEGIN
  ALTER TABLE public.mindmap_reminders
    ADD CONSTRAINT chk_days_of_week_len
    CHECK (days_of_week IS NULL OR array_length(days_of_week, 1) <= 7);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- topics_discussed: free-form text labels
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'mindmap_therapy_sessions'
      AND column_name = 'topics_discussed'
      AND data_type = 'ARRAY'
      AND udt_name NOT IN ('_text')
  ) THEN
    ALTER TABLE public.mindmap_therapy_sessions
      ALTER COLUMN topics_discussed TYPE text[] USING topics_discussed::text[];
  END IF;
END;
$$;

ALTER TABLE public.mindmap_therapy_sessions
  ALTER COLUMN topics_discussed SET DEFAULT '{}'::text[];


COMMIT;
