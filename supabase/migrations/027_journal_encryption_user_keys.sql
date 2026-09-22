-- ============================================================
-- Migration 027: per-user wrapped Data Encryption Keys for journals
-- ============================================================
-- Implements the storage half of ADR-001 (envelope encryption for journal
-- body). Each user has one active DEK; the DEK plaintext exists only in
-- the app process during encrypt/decrypt, and is wrapped by the master
-- key (JOURNAL_ENCRYPTION_MASTER_KEY, held in Vercel env). This table
-- stores the wrapped ciphertext + the wrapping metadata.
--
-- mindmap_journal_entries.encryption_key_id (added in migration 002)
-- references mindmap_journal_user_keys.id via app logic, not a hard FK:
-- rotating a user's key produces a new row here, but old journal rows
-- keep their historical key_id so they still decrypt against the DEK that
-- wrapped them. A hard FK would either need cascade (destroys ciphertext
-- readability) or restrict (blocks rotation).
--
-- Writes are service-role only (the encrypt helper runs server-side and
-- has the master key; the browser never touches raw DEKs). Reads: a user
-- can read their own wrapped key rows to enumerate/rotate; nothing else.
-- ============================================================

BEGIN;

CREATE TABLE IF NOT EXISTS public.mindmap_journal_user_keys (
  id text PRIMARY KEY,                          -- opaque key id referenced by mindmap_journal_entries.encryption_key_id
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  wrapped_dek bytea NOT NULL,                   -- DEK ciphertext produced by wrapping under the master key
  wrap_algo text NOT NULL
    CHECK (wrap_algo = ANY (ARRAY['aes-256-gcm'])),
  wrap_master_key_version text NOT NULL,        -- which master-key version wrapped this DEK; identifies rotation era
  wrap_iv bytea NOT NULL,                       -- 12-byte IV used for the wrap step
  wrap_auth_tag bytea NOT NULL,                 -- 16-byte GCM auth tag for the wrap step
  is_active boolean NOT NULL DEFAULT true,      -- exactly one row per user with is_active=true; historical keys are kept for decrypt
  created_at timestamptz NOT NULL DEFAULT now(),
  rotated_at timestamptz                        -- set when the row is superseded by a newer active DEK
);

COMMENT ON TABLE  public.mindmap_journal_user_keys IS
  'Per-user wrapped Data Encryption Keys (envelope encryption, ADR-001). One active row per user; rotated keys are kept so historical journal rows remain decryptable.';
COMMENT ON COLUMN public.mindmap_journal_user_keys.id IS
  'App-generated opaque key id (e.g. `k_<uuid>`). Referenced by mindmap_journal_entries.encryption_key_id.';
COMMENT ON COLUMN public.mindmap_journal_user_keys.wrapped_dek IS
  'DEK ciphertext. Unwrap with JOURNAL_ENCRYPTION_MASTER_KEY[wrap_master_key_version] using wrap_algo + wrap_iv + wrap_auth_tag.';

-- Fast lookups: (user_id, is_active=true) is the read path for encrypt;
-- (id) is the read path for decrypt of an existing row.
CREATE UNIQUE INDEX IF NOT EXISTS uq_mindmap_journal_user_keys_active
  ON public.mindmap_journal_user_keys (user_id)
  WHERE is_active = true;

CREATE INDEX IF NOT EXISTS idx_mindmap_journal_user_keys_user
  ON public.mindmap_journal_user_keys (user_id);

-- RLS
ALTER TABLE public.mindmap_journal_user_keys ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS journal_user_keys_owner_read ON public.mindmap_journal_user_keys;
CREATE POLICY journal_user_keys_owner_read
  ON public.mindmap_journal_user_keys
  FOR SELECT
  TO authenticated
  USING (user_id = auth.uid());

-- No INSERT/UPDATE/DELETE policies for authenticated: writes are service-role
-- only (encrypt/rotate/backfill helpers run server-side under the service
-- key). service_role bypasses RLS by default so no explicit policy needed.

COMMIT;
