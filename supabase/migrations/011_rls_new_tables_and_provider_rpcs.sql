-- ============================================================
-- Migration 007: RLS for migration-005 tables + provider RPC layer
-- ============================================================
-- - Enables RLS on every table introduced in 005
-- - Adds owner-only policies for patient-owned tables
-- - Adds append-only policies for security_audit_events
-- - Adds public-read policy for legal_documents
-- - Introduces SECURITY DEFINER RPCs for provider-side reads with
--   audited logging into security_audit_events.
-- ============================================================

BEGIN;

-- ============================================================
-- 1) LEGAL DOCUMENTS — readable by anyone (including anon)
-- ============================================================

ALTER TABLE public.legal_documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone reads active legal documents"
  ON public.legal_documents;
CREATE POLICY "Anyone reads active legal documents"
  ON public.legal_documents FOR SELECT
  USING (is_active = true);

-- No INSERT/UPDATE/DELETE policies → only service_role can mutate.


-- ============================================================
-- 2) USER PRIVACY SETTINGS — owner only
-- ============================================================

ALTER TABLE public.user_privacy_settings ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users manage own privacy settings"
  ON public.user_privacy_settings;
CREATE POLICY "Users manage own privacy settings"
  ON public.user_privacy_settings FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);


-- ============================================================
-- 3) ATTACHMENTS — owner only
-- ============================================================

ALTER TABLE public.mindmap_attachments ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users manage own attachments"
  ON public.mindmap_attachments;
CREATE POLICY "Users manage own attachments"
  ON public.mindmap_attachments FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);


-- ============================================================
-- 4) SECURITY AUDIT EVENTS — append-only, target user reads own
-- ============================================================

ALTER TABLE public.security_audit_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users read events targeting them"
  ON public.security_audit_events;
CREATE POLICY "Users read events targeting them"
  ON public.security_audit_events FOR SELECT
  USING (auth.uid() = target_user_id OR auth.uid() = actor_user_id);

DROP POLICY IF EXISTS "Authenticated users insert audit events"
  ON public.security_audit_events;
CREATE POLICY "Authenticated users insert audit events"
  ON public.security_audit_events FOR INSERT
  WITH CHECK (auth.uid() = actor_user_id OR auth.uid() IS NOT NULL);

-- Update/delete are blocked at trigger level (fn_security_audit_immutable).


-- ============================================================
-- 5) DEVICE PUSH TOKENS — owner only
-- ============================================================

ALTER TABLE public.device_push_tokens ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users manage own device tokens"
  ON public.device_push_tokens;
CREATE POLICY "Users manage own device tokens"
  ON public.device_push_tokens FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);


-- ============================================================
-- 6) NOTIFICATION DELIVERY LOG — read own, system writes
-- ============================================================

ALTER TABLE public.notification_delivery_log ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users read own notification log"
  ON public.notification_delivery_log;
CREATE POLICY "Users read own notification log"
  ON public.notification_delivery_log FOR SELECT
  USING (auth.uid() = user_id);

-- No INSERT/UPDATE/DELETE policies → service_role only.


-- ============================================================
-- 7) REMINDER OCCURRENCES — owner only
-- ============================================================

ALTER TABLE public.reminder_occurrences ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users manage own reminder occurrences"
  ON public.reminder_occurrences;
CREATE POLICY "Users manage own reminder occurrences"
  ON public.reminder_occurrences FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);


-- ============================================================
-- 8) PROVIDER ACCESS RPCS
--    Security-definer functions enforce share scope server-side
--    and emit security_audit_events rows on every read.
-- ============================================================

-- Helper: assert the calling provider has an active share for
-- (patient, resource_type, date) covering the requested window.
CREATE OR REPLACE FUNCTION fn_provider_can_read(
  p_provider_user_id uuid,
  p_patient_user_id uuid,
  p_resource_type text,
  p_date_from date,
  p_date_to date
) RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1
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
  );
$$;

REVOKE ALL ON FUNCTION fn_provider_can_read(uuid, uuid, text, date, date) FROM public;
GRANT EXECUTE ON FUNCTION fn_provider_can_read(uuid, uuid, text, date, date) TO authenticated;


-- Audit helper
CREATE OR REPLACE FUNCTION fn_log_provider_access(
  p_provider_user_id uuid,
  p_patient_user_id uuid,
  p_resource_type text,
  p_record_count integer
) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  INSERT INTO security_audit_events (
    actor_user_id, target_user_id, event_type, table_name, metadata
  ) VALUES (
    p_provider_user_id,
    p_patient_user_id,
    'provider_viewed_patient_data',
    p_resource_type,
    jsonb_build_object('record_count', p_record_count)
  );
END;
$$;

REVOKE ALL ON FUNCTION fn_log_provider_access(uuid, uuid, text, integer) FROM public;
GRANT EXECUTE ON FUNCTION fn_log_provider_access(uuid, uuid, text, integer) TO authenticated;


-- RPC: provider-readable entries within share window
CREATE OR REPLACE FUNCTION rpc_provider_get_entries(
  p_patient_user_id uuid,
  p_date_from date,
  p_date_to date
) RETURNS SETOF mindmap_entries
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_count integer;
BEGIN
  IF NOT fn_provider_can_read(auth.uid(), p_patient_user_id, 'entries', p_date_from, p_date_to) THEN
    RAISE EXCEPTION 'access_denied' USING ERRCODE = '42501';
  END IF;

  RETURN QUERY
    SELECT *
    FROM mindmap_entries
    WHERE user_id = p_patient_user_id
      AND entry_date BETWEEN p_date_from AND p_date_to
      AND deleted_at IS NULL
    ORDER BY entry_date DESC;

  GET DIAGNOSTICS v_count = ROW_COUNT;
  PERFORM fn_log_provider_access(auth.uid(), p_patient_user_id, 'entries', v_count);
END;
$$;

REVOKE ALL ON FUNCTION rpc_provider_get_entries(uuid, date, date) FROM public;
GRANT EXECUTE ON FUNCTION rpc_provider_get_entries(uuid, date, date) TO authenticated;


-- RPC: provider-readable insights
CREATE OR REPLACE FUNCTION rpc_provider_get_insights(
  p_patient_user_id uuid,
  p_date_from date,
  p_date_to date
) RETURNS SETOF mindmap_insights
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
    SELECT *
    FROM mindmap_insights
    WHERE user_id = p_patient_user_id
      AND computed_at::date BETWEEN p_date_from AND p_date_to
    ORDER BY computed_at DESC;

  GET DIAGNOSTICS v_count = ROW_COUNT;
  PERFORM fn_log_provider_access(auth.uid(), p_patient_user_id, 'insights', v_count);
END;
$$;

REVOKE ALL ON FUNCTION rpc_provider_get_insights(uuid, date, date) FROM public;
GRANT EXECUTE ON FUNCTION rpc_provider_get_insights(uuid, date, date) TO authenticated;


-- RPC: provider-readable journal metadata (titles, tags, never content)
CREATE OR REPLACE FUNCTION rpc_provider_get_journal_metadata(
  p_patient_user_id uuid,
  p_date_from date,
  p_date_to date
) RETURNS TABLE (
  id uuid,
  entry_date date,
  entry_time time,
  title text,
  mood_tags text[],
  created_at timestamptz
)
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_count integer;
BEGIN
  IF NOT fn_provider_can_read(auth.uid(), p_patient_user_id, 'journal', p_date_from, p_date_to) THEN
    RAISE EXCEPTION 'access_denied' USING ERRCODE = '42501';
  END IF;

  RETURN QUERY
    SELECT
      j.id,
      j.entry_date,
      j.entry_time,
      j.title,
      j.mood_tags,
      j.created_at
    FROM mindmap_journal_entries j
    WHERE j.user_id = p_patient_user_id
      AND j.entry_date BETWEEN p_date_from AND p_date_to
      AND j.deleted_at IS NULL
    ORDER BY j.entry_date DESC, j.entry_time DESC;

  GET DIAGNOSTICS v_count = ROW_COUNT;
  PERFORM fn_log_provider_access(auth.uid(), p_patient_user_id, 'journal', v_count);
END;
$$;

REVOKE ALL ON FUNCTION rpc_provider_get_journal_metadata(uuid, date, date) FROM public;
GRANT EXECUTE ON FUNCTION rpc_provider_get_journal_metadata(uuid, date, date) TO authenticated;


COMMIT;
