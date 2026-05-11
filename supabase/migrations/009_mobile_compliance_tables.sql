-- ============================================================
-- Migration 005: Mobile + Compliance Infrastructure
-- Adds tables required for store submission, push notifications,
-- attachments, broader auditing, and versioned legal documents.
-- All operations are additive and idempotent.
-- ============================================================

BEGIN;

-- ============================================================
-- 1) LEGAL DOCUMENT REGISTRY
--    Versioned source of truth for terms, privacy, disclaimers.
--    consent_records reference (document_type, version) pairs.
-- ============================================================

CREATE TABLE IF NOT EXISTS public.legal_documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_type text NOT NULL CHECK (document_type IN (
    'terms_of_service',
    'privacy_policy',
    'data_sharing_notice',
    'medical_disclaimer'
  )),
  version text NOT NULL,
  title text NOT NULL,
  body_url text NOT NULL,
  effective_at timestamptz NOT NULL,
  retired_at timestamptz,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (document_type, version)
);

CREATE INDEX IF NOT EXISTS idx_legal_documents_type_active
  ON public.legal_documents (document_type, is_active, effective_at DESC);

COMMENT ON TABLE public.legal_documents IS
  'Versioned registry of public legal documents. Each consent record cites a (document_type, version) pair.';


-- ============================================================
-- 2) USER PRIVACY SETTINGS
--    User-controlled toggles, distinct from raw consent log.
-- ============================================================

CREATE TABLE IF NOT EXISTS public.user_privacy_settings (
  user_id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  analytics_opt_in boolean NOT NULL DEFAULT false,
  email_notifications_opt_in boolean NOT NULL DEFAULT false,
  push_notifications_opt_in boolean NOT NULL DEFAULT false,
  provider_sharing_enabled boolean NOT NULL DEFAULT false,
  allow_ai_insights boolean NOT NULL DEFAULT true,
  allow_sensitive_journal_processing boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

DROP TRIGGER IF EXISTS trg_updated_at_user_privacy_settings ON public.user_privacy_settings;
CREATE TRIGGER trg_updated_at_user_privacy_settings
  BEFORE UPDATE ON public.user_privacy_settings
  FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();


-- ============================================================
-- 3) ATTACHMENT METADATA
--    Pointer rows for objects stored in Supabase Storage.
--    Storage path convention: user/{user_id}/mindmap/{resource}/{id}/{filename}
-- ============================================================

CREATE TABLE IF NOT EXISTS public.mindmap_attachments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  related_table text NOT NULL CHECK (related_table IN (
    'mindmap_entries',
    'mindmap_journal_entries',
    'mindmap_therapy_sessions',
    'mindmap_data_exports'
  )),
  related_id uuid,
  bucket_id text NOT NULL DEFAULT 'mindmap-private',
  storage_path text NOT NULL,
  filename text NOT NULL,
  mime_type text,
  size_bytes bigint,
  checksum_sha256 text,
  is_sensitive boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  UNIQUE (bucket_id, storage_path)
);

CREATE INDEX IF NOT EXISTS idx_attachments_user
  ON public.mindmap_attachments (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_attachments_related
  ON public.mindmap_attachments (related_table, related_id);


-- ============================================================
-- 4) SECURITY AUDIT EVENTS
--    Broader audit trail beyond sharing_audit_log.
-- ============================================================

CREATE TABLE IF NOT EXISTS public.security_audit_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  actor_user_id uuid REFERENCES auth.users(id) ON DELETE SET NULL,
  target_user_id uuid REFERENCES auth.users(id) ON DELETE SET NULL,
  event_type text NOT NULL,
  table_name text,
  record_id uuid,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  ip_address text,
  user_agent text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_security_audit_actor
  ON public.security_audit_events (actor_user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_security_audit_target
  ON public.security_audit_events (target_user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_security_audit_event_type
  ON public.security_audit_events (event_type, created_at DESC);

-- Immutable: deny update/delete from non-superusers
CREATE OR REPLACE FUNCTION fn_security_audit_immutable()
RETURNS TRIGGER AS $$
BEGIN
  RAISE EXCEPTION 'security_audit_events is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_security_audit_no_update ON public.security_audit_events;
CREATE TRIGGER trg_security_audit_no_update
  BEFORE UPDATE ON public.security_audit_events
  FOR EACH ROW EXECUTE FUNCTION fn_security_audit_immutable();

DROP TRIGGER IF EXISTS trg_security_audit_no_delete ON public.security_audit_events;
CREATE TRIGGER trg_security_audit_no_delete
  BEFORE DELETE ON public.security_audit_events
  FOR EACH ROW EXECUTE FUNCTION fn_security_audit_immutable();


-- ============================================================
-- 5) DEVICE PUSH TOKENS
--    APNs / FCM / Expo / web-push registration per user device.
-- ============================================================

CREATE TABLE IF NOT EXISTS public.device_push_tokens (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  platform text NOT NULL CHECK (platform IN ('ios', 'android', 'web')),
  provider text NOT NULL CHECK (provider IN ('apns', 'fcm', 'expo', 'web_push')),
  token text NOT NULL,
  device_id text,
  app_version text,
  os_version text,
  is_active boolean NOT NULL DEFAULT true,
  last_seen_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (provider, token)
);

CREATE INDEX IF NOT EXISTS idx_device_push_tokens_user_active
  ON public.device_push_tokens (user_id, is_active);

DROP TRIGGER IF EXISTS trg_updated_at_device_push_tokens ON public.device_push_tokens;
CREATE TRIGGER trg_updated_at_device_push_tokens
  BEFORE UPDATE ON public.device_push_tokens
  FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();


-- ============================================================
-- 6) NOTIFICATION DELIVERY LOG
-- ============================================================

CREATE TABLE IF NOT EXISTS public.notification_delivery_log (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  reminder_id uuid REFERENCES public.mindmap_reminders(id) ON DELETE SET NULL,
  device_token_id uuid REFERENCES public.device_push_tokens(id) ON DELETE SET NULL,
  channel text NOT NULL CHECK (channel IN ('push', 'email', 'local', 'sms')),
  title text NOT NULL,
  body text,
  status text NOT NULL DEFAULT 'queued' CHECK (status IN (
    'queued', 'sent', 'delivered', 'failed', 'cancelled'
  )),
  provider_message_id text,
  error_message text,
  scheduled_for timestamptz,
  sent_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_notif_log_user_created
  ON public.notification_delivery_log (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_notif_log_status_scheduled
  ON public.notification_delivery_log (status, scheduled_for)
  WHERE status IN ('queued', 'sent');


-- ============================================================
-- 7) REMINDER OCCURRENCES
--    Concrete scheduled instances of mindmap_reminders.
-- ============================================================

CREATE TABLE IF NOT EXISTS public.reminder_occurrences (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  reminder_id uuid NOT NULL REFERENCES public.mindmap_reminders(id) ON DELETE CASCADE,
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  scheduled_for timestamptz NOT NULL,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN (
    'pending', 'triggered', 'completed', 'skipped', 'missed', 'cancelled'
  )),
  completed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (reminder_id, scheduled_for)
);

CREATE INDEX IF NOT EXISTS idx_reminder_occ_user_scheduled
  ON public.reminder_occurrences (user_id, scheduled_for);

CREATE INDEX IF NOT EXISTS idx_reminder_occ_status_scheduled
  ON public.reminder_occurrences (status, scheduled_for)
  WHERE status = 'pending';


-- ============================================================
-- 8) PROVIDER CLIENT UNIQUENESS + LIFECYCLE
-- ============================================================

DO $$ BEGIN
  ALTER TABLE public.provider_clients
    ADD CONSTRAINT uq_provider_clients_org_patient UNIQUE (org_id, patient_user_id);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE public.provider_clients
  ADD COLUMN IF NOT EXISTS invite_expires_at timestamptz,
  ADD COLUMN IF NOT EXISTS accepted_at timestamptz,
  ADD COLUMN IF NOT EXISTS revoked_at timestamptz;


-- ============================================================
-- 9) DELETION PROCESSING METADATA
-- ============================================================

ALTER TABLE public.data_deletion_requests
  ADD COLUMN IF NOT EXISTS confirmation_token_hash text,
  ADD COLUMN IF NOT EXISTS confirmed_at timestamptz,
  ADD COLUMN IF NOT EXISTS failure_reason text,
  ADD COLUMN IF NOT EXISTS retained_metadata jsonb NOT NULL DEFAULT '{}'::jsonb;


-- ============================================================
-- 10) INSIGHT CLAIM-SAFETY COLUMNS
-- ============================================================

ALTER TABLE public.mindmap_insights
  ADD COLUMN IF NOT EXISTS model_version text,
  ADD COLUMN IF NOT EXISTS confidence numeric
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
  ADD COLUMN IF NOT EXISTS disclaimer text
    DEFAULT 'This insight is for self-tracking and wellness reflection only. It is not a diagnosis, treatment, or medical advice.';


COMMIT;
