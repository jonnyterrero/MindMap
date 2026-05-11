-- Migration 006: Remaining guardrails (constraints, indexes, journal safety)
-- Run in Supabase SQL Editor
-- https://supabase.com/dashboard/project/zunpccwjghwpiljwwjpv/sql

BEGIN;

-- =========================================================================
-- 1) UNIQUE PARTIAL INDEX: prevent duplicate active shares
--    Without this, a patient could accidentally create multiple active
--    shares for the same resource to the same provider.
-- =========================================================================

DROP INDEX IF EXISTS public.idx_data_shares_active_unique;

CREATE UNIQUE INDEX idx_data_shares_active_unique
  ON public.data_shares (patient_user_id, provider_client_id, resource_type, detail_level)
  WHERE (is_active = true);

-- =========================================================================
-- 2) MISSING INDEXES (3 of them)
-- =========================================================================

-- Routine analytics: "which routines get completed most?"
CREATE INDEX IF NOT EXISTS idx_entry_routines_routine
  ON public.mindmap_entry_routines (routine_id);

-- Provider views: fast org-level client lookups
CREATE INDEX IF NOT EXISTS idx_provider_clients_org
  ON public.provider_clients (org_id);

-- Provider views: fast share lookups by provider_client
CREATE INDEX IF NOT EXISTS idx_data_shares_provider_client_active
  ON public.data_shares (provider_client_id, is_active);

-- =========================================================================
-- 3) CONSENT: prevent duplicate consent rows per version
--    One row per (user, type, version). If a user re-consents to a new
--    version, that's a new row. But they can't spam the same version.
-- =========================================================================

ALTER TABLE public.consent_records
  DROP CONSTRAINT IF EXISTS consent_records_user_type_version_uq;

ALTER TABLE public.consent_records
  ADD CONSTRAINT consent_records_user_type_version_uq
    UNIQUE (user_id, consent_type, consent_version);

-- =========================================================================
-- 4) JOURNAL: enforce plaintext/encrypted exclusivity
--    When encryption is active, content must be cleared.
--    When no encryption, body_encrypted must be null.
--    This prevents "encrypted but also plaintext" leaks.
-- =========================================================================

-- First, allow content to be nullable (it's currently NOT NULL)
ALTER TABLE public.mindmap_journal_entries
  ALTER COLUMN content DROP NOT NULL;

-- Add the exclusivity check
ALTER TABLE public.mindmap_journal_entries
  DROP CONSTRAINT IF EXISTS journal_encryption_exclusivity;

ALTER TABLE public.mindmap_journal_entries
  ADD CONSTRAINT journal_encryption_exclusivity
    CHECK (
      (encryption_algo = 'none' AND body_encrypted IS NULL)
      OR
      (encryption_algo != 'none' AND content IS NULL AND body_encrypted IS NOT NULL)
    );

-- =========================================================================
-- 5) DATA SHARES: change CASCADE → RESTRICT on provider_client FK
--    Prevents a provider from deleting a client row and silently
--    nuking all sharing grants. Force explicit revocation first.
-- =========================================================================

ALTER TABLE public.data_shares
  DROP CONSTRAINT data_shares_provider_client_id_fkey,
  ADD CONSTRAINT data_shares_provider_client_id_fkey
    FOREIGN KEY (provider_client_id)
    REFERENCES public.provider_clients(id)
    ON DELETE RESTRICT;

COMMIT;
