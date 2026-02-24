-- Migration: Add patient-controlled sharing model
-- Run this in the Supabase SQL Editor
-- https://supabase.com/dashboard/project/zunpccwjghwpiljwwjpv/sql

BEGIN;

-- Provider organizations (clinic, practice, solo therapist, etc.)
CREATE TABLE IF NOT EXISTS public.provider_orgs (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  name text NOT NULL,
  slug text UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (id)
);

-- Ties auth user → provider role inside an org
CREATE TABLE IF NOT EXISTS public.provider_profiles (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id),
  org_id uuid NOT NULL REFERENCES public.provider_orgs(id) ON DELETE CASCADE,
  role text NOT NULL DEFAULT 'provider'
    CHECK (role = ANY (ARRAY['owner', 'admin', 'provider', 'staff'])),
  display_name text,
  credentials text,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (id),
  UNIQUE (user_id, org_id)
);

-- Relationship: provider org ↔ patient user
CREATE TABLE IF NOT EXISTS public.provider_clients (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  org_id uuid NOT NULL REFERENCES public.provider_orgs(id) ON DELETE CASCADE,
  patient_user_id uuid NOT NULL REFERENCES auth.users(id),
  status text NOT NULL DEFAULT 'pending'
    CHECK (status = ANY (ARRAY['pending', 'active', 'paused', 'revoked'])),
  invited_by uuid REFERENCES auth.users(id),
  invite_code text UNIQUE,
  notes text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (id),
  UNIQUE (org_id, patient_user_id)
);

-- Patient-controlled data sharing grants
CREATE TABLE IF NOT EXISTS public.data_shares (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  patient_user_id uuid NOT NULL REFERENCES auth.users(id),
  provider_client_id uuid NOT NULL REFERENCES public.provider_clients(id) ON DELETE CASCADE,
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
  granted_at timestamptz NOT NULL DEFAULT now(),
  revoked_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (id)
);

-- RLS: enable but keep policies open for now (tighten before launch)
ALTER TABLE public.provider_orgs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.provider_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.provider_clients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.data_shares ENABLE ROW LEVEL SECURITY;

-- Patients can manage their own shares
CREATE POLICY "Patients manage own shares"
  ON public.data_shares
  FOR ALL
  USING (auth.uid() = patient_user_id)
  WITH CHECK (auth.uid() = patient_user_id);

-- Providers can read shares granted to their org
CREATE POLICY "Providers read granted shares"
  ON public.data_shares
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.provider_clients pc
      JOIN public.provider_profiles pp ON pp.org_id = pc.org_id
      WHERE pc.id = data_shares.provider_client_id
        AND pp.user_id = auth.uid()
        AND pp.is_active = true
        AND pc.status = 'active'
    )
  );

-- Provider profiles: users see their own
CREATE POLICY "Users see own provider profiles"
  ON public.provider_profiles
  FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Provider clients: patients see their own relationships
CREATE POLICY "Patients see own provider relationships"
  ON public.provider_clients
  FOR ALL
  USING (auth.uid() = patient_user_id)
  WITH CHECK (auth.uid() = patient_user_id);

-- Provider clients: providers see clients in their org
CREATE POLICY "Providers see org clients"
  ON public.provider_clients
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.provider_profiles pp
      WHERE pp.org_id = provider_clients.org_id
        AND pp.user_id = auth.uid()
        AND pp.is_active = true
    )
  );

-- Provider orgs: members can read their own org
CREATE POLICY "Org members read own org"
  ON public.provider_orgs
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.provider_profiles pp
      WHERE pp.org_id = provider_orgs.id
        AND pp.user_id = auth.uid()
        AND pp.is_active = true
    )
  );

-- Provider orgs: owners/admins can update
CREATE POLICY "Org owners manage org"
  ON public.provider_orgs
  FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM public.provider_profiles pp
      WHERE pp.org_id = provider_orgs.id
        AND pp.user_id = auth.uid()
        AND pp.role = ANY (ARRAY['owner', 'admin'])
        AND pp.is_active = true
    )
  )
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.provider_profiles pp
      WHERE pp.org_id = provider_orgs.id
        AND pp.user_id = auth.uid()
        AND pp.role = ANY (ARRAY['owner', 'admin'])
        AND pp.is_active = true
    )
  );

COMMIT;
