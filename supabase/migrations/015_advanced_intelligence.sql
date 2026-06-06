-- ============================================================
-- Migration 015: Advanced Intelligence Roadmap
-- Predictions, AI companion, wearables, voice, reports, crisis,
-- provider access, offline queue. All additive (IF NOT EXISTS),
-- RLS user-scoped (auth.uid() = user_id). Reconciled to real
-- table names (mindmap_weather_daily, not mindmap_weather).
-- ============================================================

BEGIN;

-- profiles.role drives provider gating (Module I)
ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS role text NOT NULL DEFAULT 'patient';
DO $$ BEGIN
  ALTER TABLE public.profiles
    ADD CONSTRAINT chk_profiles_role CHECK (role IN ('patient','provider'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ---------- Predictive engine ----------
CREATE TABLE IF NOT EXISTS public.mindmap_predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  prediction_type text NOT NULL,                 -- migraine|anxiety|mood|pain_flare
  predicted_at timestamptz NOT NULL DEFAULT now(),
  risk_score numeric(4,3) CHECK (risk_score BETWEEN 0 AND 1),
  risk_level text CHECK (risk_level IN ('low','moderate','high','critical')),
  confidence numeric(4,3),
  contributing_factors jsonb NOT NULL DEFAULT '[]'::jsonb,
  model_version text DEFAULT 'v1_rule_extended',
  acknowledged_at timestamptz,
  outcome_recorded text CHECK (outcome_recorded IN ('accurate','inaccurate')),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_predictions_user_time
  ON public.mindmap_predictions (user_id, predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_user_type_time
  ON public.mindmap_predictions (user_id, prediction_type, predicted_at DESC);

-- ---------- AI conversation companion ----------
CREATE TABLE IF NOT EXISTS public.mindmap_ai_conversations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  title text,
  context_entry_id uuid REFERENCES public.mindmap_entries(id) ON DELETE SET NULL,
  context_window jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ai_conversations_user
  ON public.mindmap_ai_conversations (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS public.mindmap_ai_messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id uuid NOT NULL REFERENCES public.mindmap_ai_conversations(id) ON DELETE CASCADE,
  role text CHECK (role IN ('user','assistant','system')),
  content text NOT NULL,
  tokens_used int,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  crisis_flagged boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ai_messages_conversation
  ON public.mindmap_ai_messages (conversation_id, created_at);

-- ---------- Wearables ----------
CREATE TABLE IF NOT EXISTS public.mindmap_wearable_sources (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  source_type text NOT NULL,                     -- apple_health|health_connect|fitbit|oura
  connected_at timestamptz NOT NULL DEFAULT now(),
  last_sync_at timestamptz,
  is_active boolean NOT NULL DEFAULT true,
  credentials_ref text,                          -- ref to Supabase Vault, never the raw token
  sync_config jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (user_id, source_type)
);
CREATE INDEX IF NOT EXISTS idx_wearable_sources_user
  ON public.mindmap_wearable_sources (user_id);

CREATE TABLE IF NOT EXISTS public.mindmap_wearable_data (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  source_id uuid REFERENCES public.mindmap_wearable_sources(id) ON DELETE SET NULL,
  metric_type text NOT NULL,                     -- hrv|sleep_score|resting_hr|steps|spo2|temperature
  value numeric NOT NULL,
  unit text,
  recorded_at timestamptz NOT NULL,
  synced_at timestamptz NOT NULL DEFAULT now(),
  raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (user_id, source_id, metric_type, recorded_at)
);
CREATE INDEX IF NOT EXISTS idx_wearable_data_user_metric_time
  ON public.mindmap_wearable_data (user_id, metric_type, recorded_at DESC);

-- ---------- Voice journaling ----------
CREATE TABLE IF NOT EXISTS public.mindmap_voice_notes (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  entry_id uuid REFERENCES public.mindmap_entries(id) ON DELETE SET NULL,
  storage_path text NOT NULL,
  duration_seconds int,
  transcript text,
  transcript_status text NOT NULL DEFAULT 'pending'
    CHECK (transcript_status IN ('pending','processing','complete','failed')),
  sentiment_score numeric(4,3),
  themes jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_voice_notes_user
  ON public.mindmap_voice_notes (user_id, created_at DESC);

-- ---------- Weekly/monthly AI reports ----------
CREATE TABLE IF NOT EXISTS public.mindmap_ai_reports (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  report_type text CHECK (report_type IN ('weekly','monthly')),
  period_start date NOT NULL,
  period_end date NOT NULL,
  summary_markdown text,
  key_insights jsonb NOT NULL DEFAULT '[]'::jsonb,
  trend_data jsonb NOT NULL DEFAULT '{}'::jsonb,
  pdf_storage_path text,
  generated_at timestamptz NOT NULL DEFAULT now(),
  delivered_at timestamptz,
  UNIQUE (user_id, report_type, period_start)
);
CREATE INDEX IF NOT EXISTS idx_ai_reports_user
  ON public.mindmap_ai_reports (user_id, period_start DESC);

-- ---------- Crisis / safety escalation ----------
CREATE TABLE IF NOT EXISTS public.mindmap_crisis_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  trigger_source text,                           -- ai_message|journal_entry|manual
  trigger_content_ref uuid,
  severity text CHECK (severity IN ('concern','moderate','critical')),
  resources_shown jsonb NOT NULL DEFAULT '[]'::jsonb,
  acknowledged_at timestamptz,
  escalated_to_provider boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_crisis_events_user
  ON public.mindmap_crisis_events (user_id, created_at DESC);

-- ---------- Provider access (simple patient<->provider grant) ----------
-- NOTE: coexists with the existing org-based provider model
-- (provider_clients / data_shares). This is the lightweight grant the
-- Provider Dashboard module reads.
CREATE TABLE IF NOT EXISTS public.mindmap_provider_access (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  provider_user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  granted_at timestamptz NOT NULL DEFAULT now(),
  revoked_at timestamptz,
  permissions jsonb NOT NULL DEFAULT '{"read_entries": false, "read_reports": true, "read_predictions": true}'::jsonb,
  UNIQUE (patient_user_id, provider_user_id)
);
CREATE INDEX IF NOT EXISTS idx_provider_access_provider
  ON public.mindmap_provider_access (provider_user_id) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_provider_access_patient
  ON public.mindmap_provider_access (patient_user_id);

-- ---------- Offline queue ----------
CREATE TABLE IF NOT EXISTS public.mindmap_offline_queue (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  operation text CHECK (operation IN ('INSERT','UPDATE','DELETE')),
  target_table text NOT NULL,
  payload jsonb NOT NULL,
  local_id text,
  status text NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending','syncing','synced','conflict','failed')),
  conflict_data jsonb,
  retry_count int NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  synced_at timestamptz
);
CREATE INDEX IF NOT EXISTS idx_offline_queue_user_status
  ON public.mindmap_offline_queue (user_id, status);

-- ---------- Weather extension: air quality + pollen ----------
ALTER TABLE public.mindmap_weather_daily
  ADD COLUMN IF NOT EXISTS aqi integer,
  ADD COLUMN IF NOT EXISTS pm25 numeric(5,2),
  ADD COLUMN IF NOT EXISTS pollen_tree integer,
  ADD COLUMN IF NOT EXISTS pollen_grass integer,
  ADD COLUMN IF NOT EXISTS pollen_weed integer,
  ADD COLUMN IF NOT EXISTS pollen_level text;
DO $$ BEGIN
  ALTER TABLE public.mindmap_weather_daily
    ADD CONSTRAINT chk_weather_pollen_level
    CHECK (pollen_level IS NULL OR pollen_level IN ('low','moderate','high','very_high'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ============================================================
-- RLS — enable + owner policies
-- ============================================================
ALTER TABLE public.mindmap_predictions        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mindmap_ai_conversations   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mindmap_ai_messages        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mindmap_wearable_sources   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mindmap_wearable_data      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mindmap_voice_notes        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mindmap_ai_reports         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mindmap_crisis_events      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mindmap_provider_access    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mindmap_offline_queue      ENABLE ROW LEVEL SECURITY;

-- Helper: standard owner policy (FOR ALL) on user_id tables
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'mindmap_predictions','mindmap_ai_conversations','mindmap_wearable_sources',
    'mindmap_wearable_data','mindmap_voice_notes','mindmap_ai_reports',
    'mindmap_crisis_events','mindmap_offline_queue'
  ] LOOP
    EXECUTE format(
      'CREATE POLICY %I ON public.%I FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id)',
      t || '_owner', t
    );
  END LOOP;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ai_messages: ownership via parent conversation
DO $$ BEGIN
  CREATE POLICY ai_messages_owner ON public.mindmap_ai_messages
    FOR ALL
    USING (EXISTS (SELECT 1 FROM public.mindmap_ai_conversations c
                   WHERE c.id = conversation_id AND c.user_id = auth.uid()))
    WITH CHECK (EXISTS (SELECT 1 FROM public.mindmap_ai_conversations c
                        WHERE c.id = conversation_id AND c.user_id = auth.uid()));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- provider_access: patient manages grants; provider can read grants to them
DO $$ BEGIN
  CREATE POLICY provider_access_select ON public.mindmap_provider_access
    FOR SELECT USING (auth.uid() = patient_user_id OR auth.uid() = provider_user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  CREATE POLICY provider_access_write ON public.mindmap_provider_access
    FOR INSERT WITH CHECK (auth.uid() = patient_user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  CREATE POLICY provider_access_update ON public.mindmap_provider_access
    FOR UPDATE USING (auth.uid() = patient_user_id) WITH CHECK (auth.uid() = patient_user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  CREATE POLICY provider_access_delete ON public.mindmap_provider_access
    FOR DELETE USING (auth.uid() = patient_user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMIT;
