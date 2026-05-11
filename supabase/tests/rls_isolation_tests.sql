-- ============================================================
-- RLS Isolation Test Suite
-- ============================================================
-- Verifies that patient-owner policies and provider RPC scoping
-- actually block cross-user reads/writes.
--
-- Run as the service_role:
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f rls_isolation_tests.sql
--
-- Each test sets the JWT via set_config('request.jwt.claims', ...)
-- to impersonate a user, then asserts visible row counts.
-- Failures RAISE EXCEPTION; success prints NOTICE 'OK: ...'.
-- ============================================================

BEGIN;

-- ============================================================
-- 0) FIXTURE SETUP
-- ============================================================

-- Two patients, one provider org, one provider, one client link, one share.
-- Cleaned up at COMMIT/ROLLBACK; this whole file is one transaction.

DO $$
DECLARE
  v_user_a uuid := '00000000-0000-0000-0000-00000000000a';
  v_user_b uuid := '00000000-0000-0000-0000-00000000000b';
  v_provider uuid := '00000000-0000-0000-0000-0000000000cc';
  v_org uuid;
  v_pc uuid;
BEGIN
  -- Bypass auth.users FK by inserting test users directly.
  INSERT INTO auth.users (id, email, instance_id, aud, role, created_at, updated_at)
  VALUES
    (v_user_a, 'a@test.local', '00000000-0000-0000-0000-000000000000', 'authenticated', 'authenticated', now(), now()),
    (v_user_b, 'b@test.local', '00000000-0000-0000-0000-000000000000', 'authenticated', 'authenticated', now(), now()),
    (v_provider, 'p@test.local', '00000000-0000-0000-0000-000000000000', 'authenticated', 'authenticated', now(), now())
  ON CONFLICT (id) DO NOTHING;

  INSERT INTO mindmap_entries (user_id, entry_date, mood_valence)
  VALUES
    (v_user_a, current_date - 1, 1),
    (v_user_a, current_date - 2, 0),
    (v_user_b, current_date - 1, -1)
  ON CONFLICT (user_id, entry_date) DO NOTHING;

  INSERT INTO mindmap_journal_entries (user_id, entry_date, content, mood_tags)
  VALUES (v_user_a, current_date - 1, 'private journal a', ARRAY['anxious']::text[])
  ON CONFLICT DO NOTHING;

  INSERT INTO provider_orgs (name, owner_user_id)
  VALUES ('Test Clinic', v_provider)
  RETURNING id INTO v_org;

  INSERT INTO provider_profiles (user_id, org_id, role)
  VALUES (v_provider, v_org, 'clinician');

  INSERT INTO provider_clients (org_id, patient_user_id, status, accepted_at)
  VALUES (v_org, v_user_a, 'active', now())
  RETURNING id INTO v_pc;

  INSERT INTO data_shares (
    patient_user_id, provider_client_id, resource_type,
    date_range_start, date_range_end, is_active
  ) VALUES
    (v_user_a, v_pc, 'entries', current_date - 30, current_date, true);

  RAISE NOTICE 'Fixtures created. user_a=% user_b=% provider=% org=% pc=%',
    v_user_a, v_user_b, v_provider, v_org, v_pc;
END;
$$;


-- ============================================================
-- Helper: impersonate a user inside this session
-- ============================================================

CREATE OR REPLACE FUNCTION pg_temp.as_user(p_user_id uuid)
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
  PERFORM set_config('request.jwt.claims',
    json_build_object('sub', p_user_id, 'role', 'authenticated')::text,
    true);
  PERFORM set_config('role', 'authenticated', true);
END;
$$;

CREATE OR REPLACE FUNCTION pg_temp.assert_eq(label text, actual int, expected int)
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
  IF actual = expected THEN
    RAISE NOTICE 'OK: % (got %)', label, actual;
  ELSE
    RAISE EXCEPTION 'FAIL: % — expected %, got %', label, expected, actual;
  END IF;
END;
$$;


-- ============================================================
-- TEST 1: User A can only see own entries
-- ============================================================

DO $$
DECLARE
  v_count int;
BEGIN
  PERFORM pg_temp.as_user('00000000-0000-0000-0000-00000000000a'::uuid);
  SELECT count(*) INTO v_count FROM mindmap_entries;
  PERFORM pg_temp.assert_eq('user_a sees only own entries', v_count, 2);
END;
$$;


-- ============================================================
-- TEST 2: User B cannot see User A's entries
-- ============================================================

DO $$
DECLARE
  v_count int;
BEGIN
  PERFORM pg_temp.as_user('00000000-0000-0000-0000-00000000000b'::uuid);
  SELECT count(*) INTO v_count
  FROM mindmap_entries
  WHERE user_id = '00000000-0000-0000-0000-00000000000a';
  PERFORM pg_temp.assert_eq('user_b cannot read user_a entries', v_count, 0);
END;
$$;


-- ============================================================
-- TEST 3: User B cannot read User A's journal
-- ============================================================

DO $$
DECLARE
  v_count int;
BEGIN
  PERFORM pg_temp.as_user('00000000-0000-0000-0000-00000000000b'::uuid);
  SELECT count(*) INTO v_count
  FROM mindmap_journal_entries
  WHERE user_id = '00000000-0000-0000-0000-00000000000a';
  PERFORM pg_temp.assert_eq('user_b cannot read user_a journal', v_count, 0);
END;
$$;


-- ============================================================
-- TEST 4: User B cannot insert into User A's entries
-- ============================================================

DO $$
DECLARE
  v_error text;
BEGIN
  PERFORM pg_temp.as_user('00000000-0000-0000-0000-00000000000b'::uuid);
  BEGIN
    INSERT INTO mindmap_entries (user_id, entry_date, mood_valence)
    VALUES ('00000000-0000-0000-0000-00000000000a', current_date, 0);
    RAISE EXCEPTION 'FAIL: user_b was able to insert into user_a entries';
  EXCEPTION WHEN insufficient_privilege OR check_violation THEN
    RAISE NOTICE 'OK: user_b blocked from inserting into user_a entries';
  WHEN OTHERS THEN
    IF SQLSTATE = '42501' OR SQLERRM LIKE '%row-level security%' THEN
      RAISE NOTICE 'OK: user_b blocked from inserting into user_a entries (RLS)';
    ELSE
      RAISE;
    END IF;
  END;
END;
$$;


-- ============================================================
-- TEST 5: Provider can read shared patient entries via RPC
-- ============================================================

DO $$
DECLARE
  v_count int;
BEGIN
  PERFORM pg_temp.as_user('00000000-0000-0000-0000-0000000000cc'::uuid);
  SELECT count(*) INTO v_count
  FROM rpc_provider_get_entries(
    '00000000-0000-0000-0000-00000000000a'::uuid,
    current_date - 30,
    current_date
  );
  PERFORM pg_temp.assert_eq('provider reads shared patient entries', v_count, 2);
END;
$$;


-- ============================================================
-- TEST 6: Provider cannot read entries for non-shared patient
-- ============================================================

DO $$
BEGIN
  PERFORM pg_temp.as_user('00000000-0000-0000-0000-0000000000cc'::uuid);
  BEGIN
    PERFORM count(*) FROM rpc_provider_get_entries(
      '00000000-0000-0000-0000-00000000000b'::uuid,
      current_date - 30,
      current_date
    );
    RAISE EXCEPTION 'FAIL: provider was able to read non-shared patient entries';
  EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE 'OK: provider blocked from non-shared patient';
  WHEN OTHERS THEN
    IF SQLERRM LIKE '%access_denied%' OR SQLSTATE = '42501' THEN
      RAISE NOTICE 'OK: provider blocked from non-shared patient';
    ELSE
      RAISE;
    END IF;
  END;
END;
$$;


-- ============================================================
-- TEST 7: Revoked share blocks provider access
-- ============================================================

DO $$
DECLARE
  v_count int;
BEGIN
  -- Service role: revoke the share
  PERFORM set_config('role', 'service_role', true);
  UPDATE data_shares
    SET is_active = false, revoked_at = now()
    WHERE patient_user_id = '00000000-0000-0000-0000-00000000000a'
      AND resource_type = 'entries';

  PERFORM pg_temp.as_user('00000000-0000-0000-0000-0000000000cc'::uuid);
  BEGIN
    SELECT count(*) INTO v_count FROM rpc_provider_get_entries(
      '00000000-0000-0000-0000-00000000000a'::uuid,
      current_date - 30,
      current_date
    );
    RAISE EXCEPTION 'FAIL: provider still reads after share revoked (count=%)', v_count;
  EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE 'OK: revoked share blocks provider read';
  WHEN OTHERS THEN
    IF SQLERRM LIKE '%access_denied%' OR SQLSTATE = '42501' THEN
      RAISE NOTICE 'OK: revoked share blocks provider read';
    ELSE
      RAISE;
    END IF;
  END;
END;
$$;


-- ============================================================
-- TEST 8: User cannot read another user's privacy settings
-- ============================================================

DO $$
DECLARE
  v_count int;
BEGIN
  PERFORM set_config('role', 'service_role', true);
  INSERT INTO user_privacy_settings (user_id, push_notifications_opt_in)
  VALUES ('00000000-0000-0000-0000-00000000000a', true)
  ON CONFLICT (user_id) DO NOTHING;

  PERFORM pg_temp.as_user('00000000-0000-0000-0000-00000000000b'::uuid);
  SELECT count(*) INTO v_count
  FROM user_privacy_settings
  WHERE user_id = '00000000-0000-0000-0000-00000000000a';
  PERFORM pg_temp.assert_eq('user_b cannot read user_a privacy settings', v_count, 0);
END;
$$;


-- ============================================================
-- TEST 9: Device push tokens are owner-only
-- ============================================================

DO $$
DECLARE
  v_count int;
BEGIN
  PERFORM set_config('role', 'service_role', true);
  INSERT INTO device_push_tokens (user_id, platform, provider, token)
  VALUES ('00000000-0000-0000-0000-00000000000a', 'ios', 'apns', 'token-a-123')
  ON CONFLICT (provider, token) DO NOTHING;

  PERFORM pg_temp.as_user('00000000-0000-0000-0000-00000000000b'::uuid);
  SELECT count(*) INTO v_count FROM device_push_tokens
  WHERE user_id = '00000000-0000-0000-0000-00000000000a';
  PERFORM pg_temp.assert_eq('user_b cannot read user_a push tokens', v_count, 0);
END;
$$;


-- ============================================================
-- TEST 10: legal_documents readable to anon
-- ============================================================

DO $$
DECLARE
  v_count int;
BEGIN
  PERFORM set_config('role', 'service_role', true);
  INSERT INTO legal_documents (document_type, version, title, body_url, effective_at)
  VALUES ('privacy_policy', '2026-05-10', 'Privacy Policy', '/privacy', now())
  ON CONFLICT (document_type, version) DO NOTHING;

  PERFORM set_config('role', 'anon', true);
  SELECT count(*) INTO v_count FROM legal_documents WHERE is_active = true;
  IF v_count < 1 THEN
    RAISE EXCEPTION 'FAIL: anon could not read active legal_documents';
  END IF;
  RAISE NOTICE 'OK: anon reads active legal documents (count=%)', v_count;
END;
$$;


-- ============================================================
-- RESULT
-- ============================================================

ROLLBACK;

-- All tests passed if no FAIL exceptions were raised.
