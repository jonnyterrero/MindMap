-- ============================================================
-- Migration 023: RLS policy fixes — three verified vulnerabilities
-- ============================================================
-- 1) security_audit_events INSERT policy was a no-op, letting any
--    authenticated user forge audit rows under another user's identity.
-- 2) rpc_provider_get_entries / rpc_provider_get_insights returned
--    SELECT * (including free-text notes) and never consulted
--    data_shares.detail_level.
-- 3) profiles_provider_read exposed the whole profiles row — including
--    weather_lat / weather_lon home coordinates — to any provider
--    holding any active grant.
--
-- Idempotent: safe to re-run.
-- ============================================================

BEGIN;

-- ============================================================
-- 1) AUDIT-LOG IMPERSONATION (introduced in migration 011)
-- ============================================================
-- The old policy read:
--     WITH CHECK (auth.uid() = actor_user_id OR auth.uid() IS NOT NULL)
-- The second disjunct is true for every authenticated request, so the
-- first was never evaluated: any signed-in user could insert an audit
-- event claiming any actor_user_id, forging the security audit trail or
-- attributing their own actions to someone else.
--
-- Note: fn_log_provider_access is SECURITY DEFINER owned by postgres and
-- security_audit_events does not FORCE row level security, so the
-- provider-access audit writes bypass RLS and are unaffected by this
-- tightening. Application code never inserts audit rows directly.

DROP POLICY IF EXISTS "Authenticated users insert audit events"
  ON public.security_audit_events;
DROP POLICY IF EXISTS "Users insert audit events as themselves"
  ON public.security_audit_events;

CREATE POLICY "Users insert audit events as themselves"
  ON public.security_audit_events FOR INSERT
  WITH CHECK (auth.uid() = actor_user_id);


-- ============================================================
-- 2) PROVIDER RPC DATA LEAK (introduced in migration 011)
-- ============================================================
-- Migration 004 defined detail_level on data_shares:
--   summary  = aggregates/trends only (no row-level entries)
--   no_notes = structured fields, excludes free-text notes
--   full     = everything including notes (explicit patient consent)
-- The RPCs ignored it entirely and returned whole table rows.

-- Resolve the highest detail_level granted for a (provider, patient,
-- resource, window). Predicate is deliberately identical to
-- fn_provider_can_read so the two never disagree about which shares
-- apply; this function only decides how much of a permitted read is
-- visible, never whether the read is permitted.
CREATE OR REPLACE FUNCTION public.fn_provider_detail_level(
  p_provider_user_id uuid,
  p_patient_user_id uuid,
  p_resource_type text,
  p_date_from date,
  p_date_to date
) RETURNS text
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = public
AS $$
  SELECT ds.detail_level
  FROM data_shares ds
  JOIN provider_clients pc ON pc.id = ds.provider_client_id
  JOIN provider_profiles pp ON pp.org_id = pc.org_id
  WHERE pp.user_id = p_provider_user_id
    AND pc.patient_user_id = p_patient_user_id
    AND pc.status = 'active'
    AND ds.is_active = true
    AND ds.resource_type = p_resource_type
    AND (ds.date_range_start IS NULL OR ds.date_range_start <= COALESCE(p_date_to, current_date))
    AND (ds.date_range_end IS NULL OR ds.date_range_end >= COALESCE(p_date_from, current_date))
  ORDER BY CASE ds.detail_level
             WHEN 'full' THEN 3
             WHEN 'no_notes' THEN 2
             ELSE 1
           END DESC
  LIMIT 1;
$$;

REVOKE ALL ON FUNCTION public.fn_provider_detail_level(uuid, uuid, text, date, date) FROM public;
GRANT EXECUTE ON FUNCTION public.fn_provider_detail_level(uuid, uuid, text, date, date) TO authenticated;


-- Entries: explicit column list; notes only at detail_level 'full';
-- no row-level entries at all under a 'summary' share.
-- Return type changes from SETOF mindmap_entries, so the old function
-- must be dropped rather than replaced.
DROP FUNCTION IF EXISTS public.rpc_provider_get_entries(uuid, date, date);

CREATE FUNCTION public.rpc_provider_get_entries(
  p_patient_user_id uuid,
  p_date_from date,
  p_date_to date
) RETURNS TABLE (
  entry_date date,
  sleep_minutes integer,
  sleep_quality integer,
  bed_time time,
  wake_time time,
  hrv integer,
  mood_valence integer,
  anxiety integer,
  depression integer,
  mania integer,
  focus integer,
  productivity integer,
  therapy_minutes integer,
  outside_minutes integer,
  migraine boolean,
  migraine_intensity integer,
  migraine_aura boolean,
  mindmap_score smallint,
  notes text
)
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_count  integer;
  v_detail text;
BEGIN
  IF NOT fn_provider_can_read(auth.uid(), p_patient_user_id, 'entries', p_date_from, p_date_to) THEN
    RAISE EXCEPTION 'access_denied' USING ERRCODE = '42501';
  END IF;

  v_detail := fn_provider_detail_level(auth.uid(), p_patient_user_id, 'entries', p_date_from, p_date_to);

  -- A 'summary' share conveys aggregates only. Matches migration 004,
  -- where v_shared_entries_no_notes requires detail_level IN ('no_notes','full').
  IF v_detail IS NULL OR v_detail = 'summary' THEN
    RAISE EXCEPTION 'access_denied_detail_level' USING ERRCODE = '42501';
  END IF;

  RETURN QUERY
    SELECT
      e.entry_date,
      e.sleep_minutes,
      e.sleep_quality,
      e.bed_time,
      e.wake_time,
      e.hrv,
      e.mood_valence,
      e.anxiety,
      e.depression,
      e.mania,
      e.focus,
      e.productivity,
      e.therapy_minutes,
      e.outside_minutes,
      e.migraine,
      e.migraine_intensity,
      e.migraine_aura,
      e.mindmap_score,
      CASE WHEN v_detail = 'full' THEN e.notes ELSE NULL::text END
    FROM mindmap_entries e
    WHERE e.user_id = p_patient_user_id
      AND e.entry_date BETWEEN p_date_from AND p_date_to
      AND e.deleted_at IS NULL
    ORDER BY e.entry_date DESC;

  GET DIAGNOSTICS v_count = ROW_COUNT;
  PERFORM fn_log_provider_access(auth.uid(), p_patient_user_id, 'entries', v_count);
END;
$$;

REVOKE ALL ON FUNCTION public.rpc_provider_get_entries(uuid, date, date) FROM public;
GRANT EXECUTE ON FUNCTION public.rpc_provider_get_entries(uuid, date, date) TO authenticated;


-- Insights: restricted to the column set already audited as
-- provider-safe by v_shared_insights in migration 004. Drops id,
-- user_id, entry_id (which links an insight to a specific entry),
-- is_acknowledged, acknowledged_at, created_at, recommendation,
-- model_version, confidence and disclaimer from the provider surface.
-- Insights are themselves aggregate summaries, so — consistent with
-- v_shared_insights, which applies no detail_level filter — they remain
-- readable under a 'summary' share.
DROP FUNCTION IF EXISTS public.rpc_provider_get_insights(uuid, date, date);

CREATE FUNCTION public.rpc_provider_get_insights(
  p_patient_user_id uuid,
  p_date_from date,
  p_date_to date
) RETURNS TABLE (
  insight_type text,
  risk_level text,
  score integer,
  reasons text[],
  signals jsonb,
  summary text,
  computed_at timestamptz
)
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_count integer;
BEGIN
  IF NOT fn_provider_can_read(auth.uid(), p_patient_user_id, 'insights', p_date_from, p_date_to) THEN
    RAISE EXCEPTION 'access_denied' USING ERRCODE = '42501';
  END IF;

  RETURN QUERY
    SELECT
      i.insight_type,
      i.risk_level,
      i.score,
      i.reasons,
      i.signals,
      i.summary,
      i.computed_at
    FROM mindmap_insights i
    WHERE i.user_id = p_patient_user_id
      AND i.computed_at::date BETWEEN p_date_from AND p_date_to
    ORDER BY i.computed_at DESC;

  GET DIAGNOSTICS v_count = ROW_COUNT;
  PERFORM fn_log_provider_access(auth.uid(), p_patient_user_id, 'insights', v_count);
END;
$$;

REVOKE ALL ON FUNCTION public.rpc_provider_get_insights(uuid, date, date) FROM public;
GRANT EXECUTE ON FUNCTION public.rpc_provider_get_insights(uuid, date, date) TO authenticated;


-- ============================================================
-- 3) GEOLOCATION EXPOSURE (introduced in migration 017)
-- ============================================================
-- profiles_provider_read granted SELECT on the whole profiles row to any
-- provider with any active grant. Postgres RLS is row-level only, so the
-- policy could not restrict columns: a provider could read weather_lat /
-- weather_lon (the patient's home coordinates), weather_label, timezone,
-- selected_focus, and every other profile column. The application only
-- ever needs id and display_name in a provider context.
--
-- Replaced with a SECURITY DEFINER accessor exposing exactly those two
-- columns, mirroring the existing lookup_provider pattern from 017.

DROP POLICY IF EXISTS profiles_provider_read ON public.profiles;

CREATE OR REPLACE FUNCTION public.rpc_provider_get_patient_profiles(
  p_patient_ids uuid[]
) RETURNS TABLE (id uuid, display_name text)
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = public
AS $$
  SELECT p.id, p.display_name
  FROM profiles p
  WHERE p.id = ANY(p_patient_ids)
    AND EXISTS (
      SELECT 1
      FROM mindmap_provider_access pa
      WHERE pa.provider_user_id = auth.uid()
        AND pa.patient_user_id = p.id
        AND pa.revoked_at IS NULL
    );
$$;

REVOKE ALL ON FUNCTION public.rpc_provider_get_patient_profiles(uuid[]) FROM public;
GRANT EXECUTE ON FUNCTION public.rpc_provider_get_patient_profiles(uuid[]) TO authenticated;

COMMENT ON FUNCTION public.rpc_provider_get_patient_profiles(uuid[]) IS
  'Provider-safe patient name lookup. Returns only id and display_name, '
  'and only for patients with an active (non-revoked) grant to the caller. '
  'Replaces the profiles_provider_read RLS policy, which leaked geo columns.';

COMMIT;
