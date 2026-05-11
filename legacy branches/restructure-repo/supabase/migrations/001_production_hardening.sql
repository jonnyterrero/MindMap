-- ============================================================
-- MindMap+ Production Hardening Migration
-- Run this in Supabase SQL Editor (or via supabase db push)
-- ============================================================
-- This migration adds:
--   1. Unique constraints (correctness)
--   2. ON DELETE CASCADE for entry children (data hygiene)
--   3. Performance indexes
--   4. Partial unique index for active shares
--   5. Journal encryption CHECK constraint
--   6. Provider-safe views
--   7. updated_at trigger automation
-- ============================================================

BEGIN;

-- ============================================================
-- 1) UNIQUE CONSTRAINTS (correctness)
-- ============================================================

-- A. One row per user per day
DO $$ BEGIN
  ALTER TABLE mindmap_entries
    ADD CONSTRAINT uq_entries_user_date UNIQUE (user_id, entry_date);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- B. One provider membership per org
DO $$ BEGIN
  ALTER TABLE provider_profiles
    ADD CONSTRAINT uq_provider_profiles_user_org UNIQUE (user_id, org_id);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- C. One consent record per type+version per user
DO $$ BEGIN
  ALTER TABLE consent_records
    ADD CONSTRAINT uq_consent_user_type_version UNIQUE (user_id, consent_type, consent_version);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;


-- ============================================================
-- 2) ON DELETE CASCADE for entry child tables (data hygiene)
-- ============================================================
-- Strategy: drop the existing FK, re-add with CASCADE.
-- This is safe because we're inside a transaction.

-- mindmap_entry_routines.entry_id → mindmap_entries.id
ALTER TABLE mindmap_entry_routines
  DROP CONSTRAINT IF EXISTS mindmap_entry_routines_entry_id_fkey;
ALTER TABLE mindmap_entry_routines
  ADD CONSTRAINT mindmap_entry_routines_entry_id_fkey
  FOREIGN KEY (entry_id) REFERENCES mindmap_entries(id) ON DELETE CASCADE;

-- mindmap_body_sensations.entry_id → mindmap_entries.id
ALTER TABLE mindmap_body_sensations
  DROP CONSTRAINT IF EXISTS mindmap_body_sensations_entry_id_fkey;
ALTER TABLE mindmap_body_sensations
  ADD CONSTRAINT mindmap_body_sensations_entry_id_fkey
  FOREIGN KEY (entry_id) REFERENCES mindmap_entries(id) ON DELETE CASCADE;

-- mindmap_meds.entry_id → mindmap_entries.id
ALTER TABLE mindmap_meds
  DROP CONSTRAINT IF EXISTS mindmap_meds_entry_id_fkey;
ALTER TABLE mindmap_meds
  ADD CONSTRAINT mindmap_meds_entry_id_fkey
  FOREIGN KEY (entry_id) REFERENCES mindmap_entries(id) ON DELETE CASCADE;

-- mindmap_trigger_events.entry_id → mindmap_entries.id
ALTER TABLE mindmap_trigger_events
  DROP CONSTRAINT IF EXISTS mindmap_trigger_events_entry_id_fkey;
ALTER TABLE mindmap_trigger_events
  ADD CONSTRAINT mindmap_trigger_events_entry_id_fkey
  FOREIGN KEY (entry_id) REFERENCES mindmap_entries(id) ON DELETE CASCADE;

-- mindmap_insights.entry_id → mindmap_entries.id
ALTER TABLE mindmap_insights
  DROP CONSTRAINT IF EXISTS mindmap_insights_entry_id_fkey;
ALTER TABLE mindmap_insights
  ADD CONSTRAINT mindmap_insights_entry_id_fkey
  FOREIGN KEY (entry_id) REFERENCES mindmap_entries(id) ON DELETE CASCADE;

-- mindmap_medication_adherence.entry_id → mindmap_entries.id (optional but recommended)
ALTER TABLE mindmap_medication_adherence
  DROP CONSTRAINT IF EXISTS mindmap_medication_adherence_entry_id_fkey;
ALTER TABLE mindmap_medication_adherence
  ADD CONSTRAINT mindmap_medication_adherence_entry_id_fkey
  FOREIGN KEY (entry_id) REFERENCES mindmap_entries(id) ON DELETE CASCADE;


-- ============================================================
-- 3) PERFORMANCE INDEXES
-- ============================================================

-- Dashboard: entries by user, most recent first
CREATE INDEX IF NOT EXISTS idx_entries_user_date_desc
  ON mindmap_entries (user_id, entry_date DESC);

-- Daily join tables: fast lookup by entry_id
CREATE INDEX IF NOT EXISTS idx_entry_routines_entry_id
  ON mindmap_entry_routines (entry_id);

CREATE INDEX IF NOT EXISTS idx_body_sensations_entry_id
  ON mindmap_body_sensations (entry_id);

CREATE INDEX IF NOT EXISTS idx_meds_entry_id
  ON mindmap_meds (entry_id);

-- Insights: user timeline (uses computed_at if it exists, else created_at)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'mindmap_insights'
      AND column_name = 'computed_at'
  ) THEN
    CREATE INDEX IF NOT EXISTS idx_insights_user_computed
      ON mindmap_insights (user_id, computed_at DESC);
  ELSE
    CREATE INDEX IF NOT EXISTS idx_insights_user_created
      ON mindmap_insights (user_id, created_at DESC);
  END IF;
END;
$$;

-- Provider system
CREATE INDEX IF NOT EXISTS idx_provider_clients_org
  ON provider_clients (org_id);

CREATE INDEX IF NOT EXISTS idx_provider_clients_patient
  ON provider_clients (patient_user_id);

CREATE INDEX IF NOT EXISTS idx_provider_profiles_org
  ON provider_profiles (org_id);

CREATE INDEX IF NOT EXISTS idx_provider_profiles_user
  ON provider_profiles (user_id);

-- Sharing: active share lookups
CREATE INDEX IF NOT EXISTS idx_data_shares_provider_active
  ON data_shares (provider_client_id, is_active);

CREATE INDEX IF NOT EXISTS idx_data_shares_patient_active
  ON data_shares (patient_user_id, is_active);


-- ============================================================
-- 4) PARTIAL UNIQUE INDEX: prevent duplicate active shares
-- ============================================================

CREATE UNIQUE INDEX IF NOT EXISTS uq_active_share_per_resource
  ON data_shares (patient_user_id, provider_client_id, resource_type, detail_level)
  WHERE is_active = true;


-- ============================================================
-- 5) JOURNAL ENCRYPTION CHECK CONSTRAINT
-- ============================================================
-- When encryption_algo != 'none', plaintext content must be NULL.
-- This prevents accidentally storing both plaintext + ciphertext.
-- Only applied if the columns exist on mindmap_journal_entries.

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'mindmap_journal_entries'
      AND column_name = 'encryption_algo'
  ) THEN
    ALTER TABLE mindmap_journal_entries
      ADD CONSTRAINT chk_journal_encryption_consistency
      CHECK (
        encryption_algo = 'none'
        OR content IS NULL
        OR content = ''
      );
  END IF;
END;
$$;


-- ============================================================
-- 6) PROVIDER-SAFE VIEWS
-- ============================================================
-- Providers query these views instead of raw patient tables.
-- Views enforce share-level access by joining through data_shares.
-- Drop existing views first (CREATE OR REPLACE cannot remove columns).

DROP VIEW IF EXISTS v_shared_entries_summary CASCADE;
DROP VIEW IF EXISTS v_shared_entries_no_notes CASCADE;
DROP VIEW IF EXISTS v_shared_entries_full CASCADE;
DROP VIEW IF EXISTS v_shared_journal_metadata CASCADE;
DROP VIEW IF EXISTS v_shared_journal_full CASCADE;
DROP VIEW IF EXISTS v_shared_insights CASCADE;
DROP VIEW IF EXISTS v_shared_medications CASCADE;

-- 6a. Summary view: mood scores, sleep, no notes/journal
CREATE VIEW v_shared_entries_summary AS
SELECT
  ds.id                     AS share_id,
  pc.org_id,
  e.user_id                 AS patient_user_id,
  e.entry_date,
  e.mood_valence,
  e.anxiety,
  e.depression,
  e.mania,
  e.focus,
  e.sleep_minutes,
  e.sleep_quality,
  e.productivity,
  e.migraine,
  e.migraine_intensity
FROM data_shares ds
JOIN provider_clients pc ON pc.id = ds.provider_client_id
JOIN mindmap_entries e   ON e.user_id = ds.patient_user_id
WHERE ds.is_active = true
  AND ds.resource_type = 'entries';

-- 6b. No-notes view: clinical data without free-text notes
CREATE VIEW v_shared_entries_no_notes AS
SELECT
  ds.id                     AS share_id,
  pc.org_id,
  e.user_id                 AS patient_user_id,
  e.entry_date,
  e.mood_valence,
  e.anxiety,
  e.depression,
  e.mania,
  e.focus,
  e.sleep_minutes,
  e.sleep_quality,
  e.bed_time,
  e.wake_time,
  e.hrv,
  e.productivity,
  e.therapy_minutes,
  e.outside_minutes,
  e.migraine,
  e.migraine_intensity,
  e.migraine_aura
FROM data_shares ds
JOIN provider_clients pc ON pc.id = ds.provider_client_id
JOIN mindmap_entries e   ON e.user_id = ds.patient_user_id
WHERE ds.is_active = true
  AND ds.resource_type = 'entries'
  AND ds.detail_level IN ('summary', 'no_notes', 'full');

-- 6c. Full view: includes notes (only when detail_level = 'full')
CREATE VIEW v_shared_entries_full AS
SELECT
  ds.id                     AS share_id,
  pc.org_id,
  e.*
FROM data_shares ds
JOIN provider_clients pc ON pc.id = ds.provider_client_id
JOIN mindmap_entries e   ON e.user_id = ds.patient_user_id
WHERE ds.is_active = true
  AND ds.resource_type = 'entries'
  AND ds.detail_level = 'full';

-- 6d. Journal metadata view: dates + mood tags, no content
CREATE VIEW v_shared_journal_metadata AS
SELECT
  ds.id                     AS share_id,
  pc.org_id,
  j.user_id                 AS patient_user_id,
  j.id                      AS journal_id,
  j.entry_date,
  j.title,
  j.mood_tags,
  j.created_at,
  j.updated_at
FROM data_shares ds
JOIN provider_clients pc           ON pc.id = ds.provider_client_id
JOIN mindmap_journal_entries j     ON j.user_id = ds.patient_user_id
WHERE ds.is_active = true
  AND ds.resource_type = 'journal';

-- 6e. Journal full view: includes content only for 'full' shares
CREATE VIEW v_shared_journal_full AS
SELECT
  ds.id                     AS share_id,
  pc.org_id,
  j.*
FROM data_shares ds
JOIN provider_clients pc           ON pc.id = ds.provider_client_id
JOIN mindmap_journal_entries j     ON j.user_id = ds.patient_user_id
WHERE ds.is_active = true
  AND ds.resource_type = 'journal'
  AND ds.detail_level = 'full';


-- ============================================================
-- 7) UPDATED_AT TRIGGER AUTOMATION
-- ============================================================

-- Reusable trigger function
CREATE OR REPLACE FUNCTION fn_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach to all tables that have updated_at
DO $$
DECLARE
  tbl TEXT;
  trigger_name TEXT;
BEGIN
  FOR tbl IN
    SELECT unnest(ARRAY[
      'mindmap_entries',
      'mindmap_entry_routines',
      'mindmap_body_sensations',
      'mindmap_meds',
      'mindmap_trigger_events',
      'mindmap_insights',
      'mindmap_medication_adherence',
      'mindmap_journal_entries',
      'mindmap_goals',
      'mindmap_reminders',
      'mindmap_routines',
      'mindmap_medication_schedule',
      'mindmap_therapy_sessions',
      'mindmap_triggers',
      'mindmap_data_exports',
      'profiles',
      'provider_clients',
      'provider_orgs',
      'provider_profiles',
      'data_shares',
      'data_deletion_requests',
      'consent_records',
      'sharing_audit_log'
    ])
  LOOP
    trigger_name := 'trg_updated_at_' || tbl;

    IF EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_schema = 'public'
        AND table_name = tbl
        AND column_name = 'updated_at'
    ) THEN
      EXECUTE format(
        'DROP TRIGGER IF EXISTS %I ON %I; '
        'CREATE TRIGGER %I BEFORE UPDATE ON %I '
        'FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();',
        trigger_name, tbl, trigger_name, tbl
      );
    END IF;
  END LOOP;
END;
$$;


COMMIT;
