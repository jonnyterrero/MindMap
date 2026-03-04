BEGIN;

-- Ensure RLS is enabled
ALTER TABLE public.provider_clients ENABLE ROW LEVEL SECURITY;

-- --------------------------------------------
-- Drop overly-permissive patient policy (if it exists)
-- --------------------------------------------
DROP POLICY IF EXISTS "Patients see own provider relationships" ON public.provider_clients;
DROP POLICY IF EXISTS "Providers see org clients" ON public.provider_clients;
DROP POLICY IF EXISTS "Patients select own provider relationships" ON public.provider_clients;
DROP POLICY IF EXISTS "Patients update own provider relationship status" ON public.provider_clients;
DROP POLICY IF EXISTS "Providers insert org clients" ON public.provider_clients;
DROP POLICY IF EXISTS "Providers update org clients" ON public.provider_clients;
DROP POLICY IF EXISTS "Org owners delete org clients" ON public.provider_clients;

-- --------------------------------------------
-- Patient SELECT: can see their own provider relationships
-- --------------------------------------------
CREATE POLICY "Patients select own provider relationships"
ON public.provider_clients
FOR SELECT
USING (auth.uid() = patient_user_id);

-- --------------------------------------------
-- Patient UPDATE: can only change status + notes (enforced by trigger below)
-- --------------------------------------------
CREATE POLICY "Patients update own provider relationship status"
ON public.provider_clients
FOR UPDATE
USING (auth.uid() = patient_user_id)
WITH CHECK (auth.uid() = patient_user_id);

-- --------------------------------------------
-- Trigger: field-level guard for patient updates
-- --------------------------------------------
CREATE OR REPLACE FUNCTION public.fn_patient_provider_client_update_guard()
RETURNS trigger AS $$
BEGIN
  IF auth.uid() = OLD.patient_user_id THEN

    IF NEW.org_id <> OLD.org_id THEN
      RAISE EXCEPTION 'Patients cannot change org_id';
    END IF;

    IF NEW.patient_user_id <> OLD.patient_user_id THEN
      RAISE EXCEPTION 'Patients cannot change patient_user_id';
    END IF;

    IF NEW.invited_by IS DISTINCT FROM OLD.invited_by THEN
      RAISE EXCEPTION 'Patients cannot change invited_by';
    END IF;

    IF NEW.invite_code IS DISTINCT FROM OLD.invite_code THEN
      RAISE EXCEPTION 'Patients cannot change invite_code';
    END IF;

    IF NEW.status NOT IN ('active','paused','revoked') THEN
      RAISE EXCEPTION 'Invalid patient status transition';
    END IF;

    IF NEW.status = 'pending' THEN
      RAISE EXCEPTION 'Patients cannot set status back to pending';
    END IF;

    IF OLD.status = 'revoked' AND NEW.status <> 'revoked' THEN
      RAISE EXCEPTION 'Revoked relationships cannot be reactivated by patient';
    END IF;

    IF NEW.created_at <> OLD.created_at THEN
      RAISE EXCEPTION 'Patients cannot change created_at';
    END IF;

  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS trg_patient_provider_client_update_guard ON public.provider_clients;
CREATE TRIGGER trg_patient_provider_client_update_guard
BEFORE UPDATE ON public.provider_clients
FOR EACH ROW
EXECUTE FUNCTION public.fn_patient_provider_client_update_guard();

-- --------------------------------------------
-- Provider SELECT: org members can see org clients
-- --------------------------------------------
CREATE POLICY "Providers see org clients"
ON public.provider_clients
FOR SELECT
USING (
  EXISTS (
    SELECT 1
    FROM public.provider_profiles pp
    WHERE pp.org_id = provider_clients.org_id
      AND pp.user_id = auth.uid()
      AND pp.is_active = true
  )
);

-- --------------------------------------------
-- Provider INSERT: org members can create/invite clients
-- --------------------------------------------
CREATE POLICY "Providers insert org clients"
ON public.provider_clients
FOR INSERT
WITH CHECK (
  EXISTS (
    SELECT 1
    FROM public.provider_profiles pp
    WHERE pp.org_id = org_id
      AND pp.user_id = auth.uid()
      AND pp.is_active = true
  )
  AND status IN ('pending','active','paused','revoked')
);

-- --------------------------------------------
-- Provider UPDATE: org members can manage clients
-- --------------------------------------------
CREATE POLICY "Providers update org clients"
ON public.provider_clients
FOR UPDATE
USING (
  EXISTS (
    SELECT 1
    FROM public.provider_profiles pp
    WHERE pp.org_id = provider_clients.org_id
      AND pp.user_id = auth.uid()
      AND pp.is_active = true
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1
    FROM public.provider_profiles pp
    WHERE pp.org_id = provider_clients.org_id
      AND pp.user_id = auth.uid()
      AND pp.is_active = true
  )
);

-- DELETE: no policies => blocked under RLS for everyone.

COMMIT;
