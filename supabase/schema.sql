-- MindMap Supabase Schema (reference only — not meant to be run directly)
-- Project: zunpccwjghwpiljwwjpv
-- Last updated: 2026-02-24

CREATE TABLE public.profiles (
  id uuid NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  display_name text,
  timezone text DEFAULT 'America/New_York'::text,
  CONSTRAINT profiles_pkey PRIMARY KEY (id),
  CONSTRAINT profiles_id_fkey FOREIGN KEY (id) REFERENCES auth.users(id)
);

CREATE TABLE public.mindmap_entries (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  entry_date date NOT NULL,
  sleep_minutes integer CHECK (sleep_minutes >= 0 AND sleep_minutes <= (24 * 60)),
  sleep_quality integer CHECK (sleep_quality >= 1 AND sleep_quality <= 5),
  bed_time time without time zone,
  wake_time time without time zone,
  hrv integer CHECK (hrv IS NULL OR hrv >= 0 AND hrv <= 400),
  mood_valence integer CHECK (mood_valence IS NULL OR mood_valence >= -3 AND mood_valence <= 3),
  anxiety integer CHECK (anxiety IS NULL OR anxiety >= 0 AND anxiety <= 10),
  depression integer CHECK (depression IS NULL OR depression >= 0 AND depression <= 10),
  mania integer CHECK (mania IS NULL OR mania >= 0 AND mania <= 10),
  focus integer CHECK (focus IS NULL OR focus >= 0 AND focus <= 10),
  productivity integer CHECK (productivity IS NULL OR productivity >= 0 AND productivity <= 100),
  therapy_minutes integer CHECK (therapy_minutes IS NULL OR therapy_minutes >= 0 AND therapy_minutes <= (24 * 60)),
  outside_minutes integer CHECK (outside_minutes IS NULL OR outside_minutes >= 0 AND outside_minutes <= (24 * 60)),
  migraine boolean NOT NULL DEFAULT false,
  migraine_intensity integer CHECK (migraine_intensity IS NULL OR migraine_intensity >= 0 AND migraine_intensity <= 10),
  migraine_aura boolean,
  notes text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT mindmap_entries_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_entries_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);

CREATE TABLE public.mindmap_body_sensations (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  entry_id uuid NOT NULL,
  body_part text NOT NULL,
  intensity integer NOT NULL CHECK (intensity >= 0 AND intensity <= 10),
  notes text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT mindmap_body_sensations_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_body_sensations_entry_id_fkey FOREIGN KEY (entry_id) REFERENCES public.mindmap_entries(id)
);

CREATE TABLE public.mindmap_meds (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  entry_id uuid NOT NULL,
  name text NOT NULL,
  dose_mg numeric,
  taken_at time without time zone,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT mindmap_meds_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_meds_entry_id_fkey FOREIGN KEY (entry_id) REFERENCES public.mindmap_entries(id)
);

CREATE TABLE public.mindmap_medication_schedule (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  name text NOT NULL,
  dosage text,
  dose_mg numeric,
  frequency text NOT NULL CHECK (frequency = ANY (ARRAY['Daily', 'Weekly', 'Monthly', 'As Needed', 'Custom'])),
  reminder_time time without time zone,
  is_active boolean NOT NULL DEFAULT true,
  start_date date,
  end_date date,
  notes text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT mindmap_medication_schedule_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_medication_schedule_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);

CREATE TABLE public.mindmap_medication_adherence (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  medication_schedule_id uuid NOT NULL,
  entry_id uuid,
  scheduled_time time without time zone NOT NULL,
  taken_at timestamp with time zone,
  was_taken boolean NOT NULL DEFAULT false,
  was_skipped boolean NOT NULL DEFAULT false,
  notes text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT mindmap_medication_adherence_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_medication_adherence_medication_schedule_id_fkey FOREIGN KEY (medication_schedule_id) REFERENCES public.mindmap_medication_schedule(id),
  CONSTRAINT mindmap_medication_adherence_entry_id_fkey FOREIGN KEY (entry_id) REFERENCES public.mindmap_entries(id)
);

CREATE TABLE public.mindmap_routines (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  name text NOT NULL,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT mindmap_routines_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_routines_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);

CREATE TABLE public.mindmap_entry_routines (
  entry_id uuid NOT NULL,
  routine_id uuid NOT NULL,
  completed boolean NOT NULL DEFAULT true,
  CONSTRAINT mindmap_entry_routines_pkey PRIMARY KEY (entry_id, routine_id),
  CONSTRAINT mindmap_entry_routines_entry_id_fkey FOREIGN KEY (entry_id) REFERENCES public.mindmap_entries(id),
  CONSTRAINT mindmap_entry_routines_routine_id_fkey FOREIGN KEY (routine_id) REFERENCES public.mindmap_routines(id)
);

CREATE TABLE public.mindmap_journal_entries (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  entry_date date NOT NULL,
  entry_time time without time zone DEFAULT CURRENT_TIME,
  title text,
  content text NOT NULL,
  mood_tags text[],
  is_private boolean NOT NULL DEFAULT true,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT mindmap_journal_entries_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_journal_entries_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);

CREATE TABLE public.mindmap_therapy_sessions (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  session_date date NOT NULL,
  session_time time without time zone,
  duration_minutes integer CHECK (duration_minutes > 0 AND duration_minutes <= (24 * 60)),
  therapist_name text,
  session_type text CHECK (session_type = ANY (ARRAY['Individual', 'Group', 'Couples', 'Family', 'Other'])),
  notes text,
  mood_before integer CHECK (mood_before IS NULL OR mood_before >= -3 AND mood_before <= 3),
  mood_after integer CHECK (mood_after IS NULL OR mood_after >= -3 AND mood_after <= 3),
  topics_discussed text[],
  homework_assigned text,
  next_session_date date,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT mindmap_therapy_sessions_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_therapy_sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);

CREATE TABLE public.mindmap_goals (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  title text NOT NULL,
  description text,
  category text CHECK (category = ANY (ARRAY['Sleep', 'Mood', 'Exercise', 'Productivity', 'Medication', 'Therapy', 'Other'])),
  target_value numeric,
  current_value numeric DEFAULT 0,
  unit text,
  target_date date,
  is_completed boolean NOT NULL DEFAULT false,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  completed_at timestamp with time zone,
  CONSTRAINT mindmap_goals_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_goals_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);

CREATE TABLE public.mindmap_triggers (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  name text NOT NULL,
  description text,
  condition_type text NOT NULL CHECK (condition_type = ANY (ARRAY['Sleep', 'Anxiety', 'Depression', 'Mania', 'Mood', 'Migraine', 'Medication', 'Custom'])),
  operator text NOT NULL CHECK (operator = ANY (ARRAY['<', '<=', '>', '>=', '=', '!='])),
  threshold_value numeric NOT NULL,
  is_active boolean NOT NULL DEFAULT true,
  alert_message text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT mindmap_triggers_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_triggers_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);

CREATE TABLE public.mindmap_trigger_events (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  trigger_id uuid NOT NULL,
  entry_id uuid,
  triggered_value numeric NOT NULL,
  triggered_at timestamp with time zone NOT NULL DEFAULT now(),
  acknowledged boolean NOT NULL DEFAULT false,
  acknowledged_at timestamp with time zone,
  CONSTRAINT mindmap_trigger_events_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_trigger_events_trigger_id_fkey FOREIGN KEY (trigger_id) REFERENCES public.mindmap_triggers(id),
  CONSTRAINT mindmap_trigger_events_entry_id_fkey FOREIGN KEY (entry_id) REFERENCES public.mindmap_entries(id)
);

CREATE TABLE public.mindmap_reminders (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  title text NOT NULL,
  description text,
  reminder_type text NOT NULL CHECK (reminder_type = ANY (ARRAY['Medication', 'Routine', 'Therapy', 'Journal', 'Check-in', 'Custom'])),
  reminder_time time without time zone NOT NULL,
  days_of_week integer[] CHECK (array_length(days_of_week, 1) <= 7),
  is_active boolean NOT NULL DEFAULT true,
  last_triggered_at timestamp with time zone,
  next_trigger_at timestamp with time zone,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT mindmap_reminders_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_reminders_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);

CREATE TABLE public.mindmap_data_exports (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  export_type text NOT NULL CHECK (export_type = ANY (ARRAY['CSV', 'JSON', 'PDF', 'Full Backup'])),
  date_range_start date,
  date_range_end date,
  file_size_bytes bigint,
  export_status text NOT NULL DEFAULT 'completed' CHECK (export_status = ANY (ARRAY['pending', 'processing', 'completed', 'failed'])),
  download_url text,
  expires_at timestamp with time zone,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT mindmap_data_exports_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_data_exports_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);

-- ============================================================================
-- SHARING MODEL
-- Patient-controlled sharing: all data_shares rows are created by/for the
-- patient. Providers can only see what a patient explicitly grants.
-- ============================================================================

-- A provider's workspace (clinic, practice, solo therapist, etc.)
CREATE TABLE public.provider_orgs (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  name text NOT NULL,
  slug text UNIQUE,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT provider_orgs_pkey PRIMARY KEY (id)
);

-- Ties an auth user to a provider role inside an org.
-- One user can belong to multiple orgs; one org can have many staff.
CREATE TABLE public.provider_profiles (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  org_id uuid NOT NULL,
  role text NOT NULL DEFAULT 'provider'
    CHECK (role = ANY (ARRAY['owner', 'admin', 'provider', 'staff'])),
  display_name text,
  credentials text,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT provider_profiles_pkey PRIMARY KEY (id),
  CONSTRAINT provider_profiles_user_org_uq UNIQUE (user_id, org_id),
  CONSTRAINT provider_profiles_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id),
  CONSTRAINT provider_profiles_org_id_fkey FOREIGN KEY (org_id) REFERENCES public.provider_orgs(id) ON DELETE CASCADE
);

-- Relationship between a provider org and a patient user.
-- Created when a patient accepts an invite or a provider sends one.
CREATE TABLE public.provider_clients (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  org_id uuid NOT NULL,
  patient_user_id uuid NOT NULL,
  status text NOT NULL DEFAULT 'pending'
    CHECK (status = ANY (ARRAY['pending', 'active', 'paused', 'revoked'])),
  invited_by uuid,
  invite_code text UNIQUE,
  notes text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT provider_clients_pkey PRIMARY KEY (id),
  CONSTRAINT provider_clients_org_patient_uq UNIQUE (org_id, patient_user_id),
  CONSTRAINT provider_clients_org_id_fkey FOREIGN KEY (org_id) REFERENCES public.provider_orgs(id) ON DELETE CASCADE,
  CONSTRAINT provider_clients_patient_user_id_fkey FOREIGN KEY (patient_user_id) REFERENCES auth.users(id),
  CONSTRAINT provider_clients_invited_by_fkey FOREIGN KEY (invited_by) REFERENCES auth.users(id)
);

-- What data a patient shares and with what scope.
-- PATIENT CONTROLS THIS — rows are created/revoked by the patient.
-- Granular: one row per resource type per provider relationship.
CREATE TABLE public.data_shares (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  patient_user_id uuid NOT NULL,
  provider_client_id uuid NOT NULL,
  resource_type text NOT NULL
    CHECK (resource_type = ANY (ARRAY[
      'entries', 'journal', 'medications', 'routines',
      'therapy_sessions', 'goals', 'triggers', 'body_sensations', 'all'
    ])),
  scope text NOT NULL DEFAULT 'read'
    CHECK (scope = ANY (ARRAY['read', 'read_write'])),
  date_range_start date,
  date_range_end date,
  is_active boolean NOT NULL DEFAULT true,
  granted_at timestamp with time zone NOT NULL DEFAULT now(),
  revoked_at timestamp with time zone,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT data_shares_pkey PRIMARY KEY (id),
  CONSTRAINT data_shares_patient_user_id_fkey FOREIGN KEY (patient_user_id) REFERENCES auth.users(id),
  CONSTRAINT data_shares_provider_client_id_fkey FOREIGN KEY (provider_client_id) REFERENCES public.provider_clients(id) ON DELETE CASCADE
);

-- Append-only audit log for all sharing activity.
-- No updates, no deletes — records are immutable for compliance.
CREATE TABLE public.sharing_audit_log (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  actor_user_id uuid NOT NULL,
  patient_user_id uuid NOT NULL,
  action text NOT NULL
    CHECK (action = ANY (ARRAY[
      'share_granted', 'share_revoked', 'share_paused', 'share_resumed',
      'data_viewed', 'data_exported',
      'client_invited', 'client_accepted', 'client_revoked'
    ])),
  resource_type text,
  provider_client_id uuid,
  data_share_id uuid,
  metadata jsonb DEFAULT '{}',
  ip_address text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT sharing_audit_log_pkey PRIMARY KEY (id),
  CONSTRAINT sharing_audit_log_actor_fkey FOREIGN KEY (actor_user_id) REFERENCES auth.users(id),
  CONSTRAINT sharing_audit_log_patient_fkey FOREIGN KEY (patient_user_id) REFERENCES auth.users(id)
);

-- ============================================================================
-- JOURNAL ENCRYPTION PREP (columns on mindmap_journal_entries)
-- ============================================================================
-- mindmap_journal_entries also has:
--   body_encrypted bytea           — ciphertext (Phase 3)
--   encryption_key_id text         — external key reference
--   encryption_algo text           — 'none' | 'aes-256-gcm' | 'xchacha20-poly1305'
--   encrypted_at timestamptz       — when encryption was applied

-- ============================================================================
-- EXPLAINABLE INSIGHTS
-- ============================================================================

CREATE TABLE public.mindmap_insights (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  entry_id uuid,
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
  acknowledged_at timestamp with time zone,
  computed_at timestamp with time zone NOT NULL DEFAULT now(),
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT mindmap_insights_pkey PRIMARY KEY (id),
  CONSTRAINT mindmap_insights_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id),
  CONSTRAINT mindmap_insights_entry_id_fkey FOREIGN KEY (entry_id) REFERENCES public.mindmap_entries(id)
);

-- ============================================================================
-- COMPLIANCE UX
-- ============================================================================

-- Append-only consent records
CREATE TABLE public.consent_records (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  consent_type text NOT NULL
    CHECK (consent_type = ANY (ARRAY[
      'terms_of_service', 'privacy_policy', 'data_sharing',
      'analytics_collection', 'email_notifications', 'push_notifications'
    ])),
  consent_version text NOT NULL,
  granted boolean NOT NULL,
  ip_address text,
  user_agent text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT consent_records_pkey PRIMARY KEY (id),
  CONSTRAINT consent_records_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);

-- User-initiated data deletion requests
CREATE TABLE public.data_deletion_requests (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  status text NOT NULL DEFAULT 'pending'
    CHECK (status = ANY (ARRAY['pending', 'processing', 'completed', 'cancelled'])),
  scope text NOT NULL DEFAULT 'all'
    CHECK (scope = ANY (ARRAY[
      'all', 'entries', 'journal', 'medications',
      'routines', 'therapy_sessions', 'goals', 'exports'
    ])),
  reason text,
  requested_at timestamp with time zone NOT NULL DEFAULT now(),
  processed_at timestamp with time zone,
  completed_at timestamp with time zone,
  processed_by text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT data_deletion_requests_pkey PRIMARY KEY (id),
  CONSTRAINT data_deletion_requests_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);
