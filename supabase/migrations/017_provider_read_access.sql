-- ============================================================
-- Migration 017: Provider read access (Module I)
-- Additive permissive SELECT policies so a provider can read a
-- patient's data ONLY when an active grant in mindmap_provider_access
-- carries the matching permission flag. DB-enforced (no service-role
-- bypass). Plus a SECURITY DEFINER lookup so a patient can validate a
-- provider code without exposing the profiles table.
-- ============================================================

BEGIN;

-- Predictions: provider read when granted read_predictions.
DO $$ BEGIN
  CREATE POLICY predictions_provider_read ON public.mindmap_predictions
    FOR SELECT USING (
      EXISTS (
        SELECT 1 FROM public.mindmap_provider_access pa
        WHERE pa.provider_user_id = auth.uid()
          AND pa.patient_user_id = mindmap_predictions.user_id
          AND pa.revoked_at IS NULL
          AND (pa.permissions->>'read_predictions')::boolean IS TRUE
      )
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Reports: provider read when granted read_reports.
DO $$ BEGIN
  CREATE POLICY reports_provider_read ON public.mindmap_ai_reports
    FOR SELECT USING (
      EXISTS (
        SELECT 1 FROM public.mindmap_provider_access pa
        WHERE pa.provider_user_id = auth.uid()
          AND pa.patient_user_id = mindmap_ai_reports.user_id
          AND pa.revoked_at IS NULL
          AND (pa.permissions->>'read_reports')::boolean IS TRUE
      )
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Daily entries: provider read when granted read_entries (for correlations).
DO $$ BEGIN
  CREATE POLICY entries_provider_read ON public.mindmap_entries
    FOR SELECT USING (
      EXISTS (
        SELECT 1 FROM public.mindmap_provider_access pa
        WHERE pa.provider_user_id = auth.uid()
          AND pa.patient_user_id = mindmap_entries.user_id
          AND pa.revoked_at IS NULL
          AND (pa.permissions->>'read_entries')::boolean IS TRUE
      )
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Profile name: provider read for any active grant (names are needed to list patients).
DO $$ BEGIN
  CREATE POLICY profiles_provider_read ON public.profiles
    FOR SELECT USING (
      EXISTS (
        SELECT 1 FROM public.mindmap_provider_access pa
        WHERE pa.provider_user_id = auth.uid()
          AND pa.patient_user_id = profiles.id
          AND pa.revoked_at IS NULL
      )
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Validate a provider code (their user id) without exposing profiles.
CREATE OR REPLACE FUNCTION public.lookup_provider(p_code uuid)
RETURNS TABLE (id uuid, display_name text)
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT p.id, p.display_name
  FROM public.profiles p
  WHERE p.id = p_code AND p.role = 'provider';
$$;

REVOKE ALL ON FUNCTION public.lookup_provider(uuid) FROM public;
GRANT EXECUTE ON FUNCTION public.lookup_provider(uuid) TO authenticated;

COMMIT;
