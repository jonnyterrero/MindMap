-- Migration 002: Compliance foundations + explainable insights
-- Run in Supabase SQL Editor
-- https://supabase.com/dashboard/project/zunpccwjghwpiljwwjpv/sql

BEGIN;

-- =========================================================================
-- 1) JOURNAL ENCRYPTION PREP
--    mindmap_entries.notes = light, unstructured notes (stays plaintext)
--    mindmap_journal_entries = sensitive free-text, prepped for encryption
-- =========================================================================

-- Add encrypted payload column + encryption metadata
-- body_encrypted will hold the ciphertext in Phase 3;
-- until then, content stays in the existing `content` column.
ALTER TABLE public.mindmap_journal_entries
  ADD COLUMN IF NOT EXISTS body_encrypted bytea,
  ADD COLUMN IF NOT EXISTS encryption_key_id text,
  ADD COLUMN IF NOT EXISTS encryption_algo text
    DEFAULT 'none'
    CHECK (encryption_algo = ANY (ARRAY['none', 'aes-256-gcm', 'xchacha20-poly1305'])),
  ADD COLUMN IF NOT EXISTS encrypted_at timestamptz;

COMMENT ON COLUMN public.mindmap_journal_entries.body_encrypted
  IS 'Encrypted journal body. When populated, `content` should be cleared.';
COMMENT ON COLUMN public.mindmap_journal_entries.encryption_key_id
  IS 'Reference to the key used for encryption (managed outside DB).';
COMMENT ON COLUMN public.mindmap_journal_entries.content
  IS 'Plaintext journal content. Will be migrated to body_encrypted in Phase 3.';

-- =========================================================================
-- 2) EXPLAINABLE INSIGHTS
--    Every computed insight is stored with its reasoning, not just a score.
--    Clinicians can see exactly why a risk was flagged.
-- =========================================================================

CREATE TABLE IF NOT EXISTS public.mindmap_insights (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id),
  entry_id uuid REFERENCES public.mindmap_entries(id),
  insight_type text NOT NULL
    CHECK (insight_type = ANY (ARRAY[
      'migraine_risk', 'mood_trend', 'sleep_alert',
      'anxiety_alert', 'medication_gap', 'routine_streak',
      'depression_trend', 'mania_alert', 'custom'
    ])),
  risk_level text NOT NULL
    CHECK (risk_level = ANY (ARRAY['low', 'moderate', 'high', 'critical'])),
  score integer CHECK (score >= 0 AND score <= 100),
  reasons text[] NOT NULL DEFAULT '{}',
  signals jsonb NOT NULL DEFAULT '{}',
  summary text NOT NULL,
  is_acknowledged boolean NOT NULL DEFAULT false,
  acknowledged_at timestamptz,
  computed_at timestamptz NOT NULL DEFAULT now(),
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (id)
);

COMMENT ON COLUMN public.mindmap_insights.reasons
  IS 'Human-readable list: ["Sleep < 6h", "Anxiety ≥ 6 yesterday"]';
COMMENT ON COLUMN public.mindmap_insights.signals
  IS 'Raw signal data: {"sleep_minutes": 300, "anxiety": 7, "migraine_count_7d": 3}';
COMMENT ON COLUMN public.mindmap_insights.summary
  IS 'One-liner: "High migraine risk due to sleep < 6h and anxiety ≥ 6 yesterday."';

ALTER TABLE public.mindmap_insights ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users read own insights" ON public.mindmap_insights;
CREATE POLICY "Users read own insights"
  ON public.mindmap_insights FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "System inserts insights" ON public.mindmap_insights;
CREATE POLICY "System inserts insights"
  ON public.mindmap_insights FOR INSERT
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users acknowledge insights" ON public.mindmap_insights;
CREATE POLICY "Users acknowledge insights"
  ON public.mindmap_insights FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Providers can read insights for patients who share with them
DROP POLICY IF EXISTS "Providers read shared patient insights" ON public.mindmap_insights;
CREATE POLICY "Providers read shared patient insights"
  ON public.mindmap_insights FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.data_shares ds
      JOIN public.provider_clients pc ON pc.id = ds.provider_client_id
      JOIN public.provider_profiles pp ON pp.org_id = pc.org_id
      WHERE ds.patient_user_id = mindmap_insights.user_id
        AND pp.user_id = auth.uid()
        AND pp.is_active = true
        AND pc.status = 'active'
        AND ds.is_active = true
        AND ds.resource_type = ANY (ARRAY['entries', 'all'])
    )
  );

-- =========================================================================
-- 3) COMPLIANCE UX TABLES
--    consent_records  — what the user agreed to and when
--    data_deletion_requests — user-initiated "delete my data" flow
--    Pairs with the existing mindmap_data_exports table for "export my data"
-- =========================================================================

-- Explicit consent records (append-only, like audit log)
CREATE TABLE IF NOT EXISTS public.consent_records (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id),
  consent_type text NOT NULL
    CHECK (consent_type = ANY (ARRAY[
      'terms_of_service', 'privacy_policy', 'data_sharing',
      'analytics_collection', 'email_notifications', 'push_notifications'
    ])),
  consent_version text NOT NULL,
  granted boolean NOT NULL,
  ip_address text,
  user_agent text,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (id)
);

COMMENT ON TABLE public.consent_records
  IS 'Append-only. Every consent grant or withdrawal is a new row.';

ALTER TABLE public.consent_records ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users read own consent" ON public.consent_records;
CREATE POLICY "Users read own consent"
  ON public.consent_records FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users insert consent" ON public.consent_records;
CREATE POLICY "Users insert consent"
  ON public.consent_records FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Immutability: no updates or deletes on consent records
DROP TRIGGER IF EXISTS no_consent_update ON public.consent_records;
DROP TRIGGER IF EXISTS no_consent_delete ON public.consent_records;

CREATE OR REPLACE FUNCTION prevent_consent_mutation()
  RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'Consent records are immutable';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER no_consent_update
  BEFORE UPDATE ON public.consent_records
  FOR EACH ROW EXECUTE FUNCTION prevent_consent_mutation();

CREATE TRIGGER no_consent_delete
  BEFORE DELETE ON public.consent_records
  FOR EACH ROW EXECUTE FUNCTION prevent_consent_mutation();

-- Data deletion requests
CREATE TABLE IF NOT EXISTS public.data_deletion_requests (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id),
  status text NOT NULL DEFAULT 'pending'
    CHECK (status = ANY (ARRAY['pending', 'processing', 'completed', 'cancelled'])),
  scope text NOT NULL DEFAULT 'all'
    CHECK (scope = ANY (ARRAY[
      'all', 'entries', 'journal', 'medications',
      'routines', 'therapy_sessions', 'goals', 'exports'
    ])),
  reason text,
  requested_at timestamptz NOT NULL DEFAULT now(),
  processed_at timestamptz,
  completed_at timestamptz,
  processed_by text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (id)
);

COMMENT ON TABLE public.data_deletion_requests
  IS 'User-initiated data deletion. Processing can be manual or automated.';

ALTER TABLE public.data_deletion_requests ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users manage own deletion requests" ON public.data_deletion_requests;
CREATE POLICY "Users manage own deletion requests"
  ON public.data_deletion_requests FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_insights_user_type
  ON public.mindmap_insights (user_id, insight_type, computed_at DESC);

CREATE INDEX IF NOT EXISTS idx_consent_user_type
  ON public.consent_records (user_id, consent_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_deletion_user_status
  ON public.data_deletion_requests (user_id, status);

COMMIT;
