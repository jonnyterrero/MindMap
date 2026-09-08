-- 026_account_deletion_cascade.sql
-- Settings → Data & privacy → deleteAccount inserts a data_deletion_requests
-- row then calls auth.admin.deleteUser. Three FKs had no ON DELETE action, so
-- the user delete failed whenever those rows existed — which they always do
-- after signup (consent), insight generation, and the audit insert itself.
-- Verified on prod 2026-09-07: DELETE FROM auth.users raised 23503 on
-- mindmap_insights_user_id_fkey.
-- Idempotent: safe to re-run.

BEGIN;

ALTER TABLE public.mindmap_insights
  DROP CONSTRAINT IF EXISTS mindmap_insights_user_id_fkey;
ALTER TABLE public.mindmap_insights
  ADD CONSTRAINT mindmap_insights_user_id_fkey
  FOREIGN KEY (user_id) REFERENCES auth.users (id) ON DELETE CASCADE;

-- Public /data-deletion page: consent rows are removed with the account.
ALTER TABLE public.consent_records
  DROP CONSTRAINT IF EXISTS consent_records_user_id_fkey;
ALTER TABLE public.consent_records
  ADD CONSTRAINT consent_records_user_id_fkey
  FOREIGN KEY (user_id) REFERENCES auth.users (id) ON DELETE CASCADE;

-- Account-deletion audit must survive the user. Null the FK; keep
-- retained_metadata.deleted_user_id as the anonymized pointer.
ALTER TABLE public.data_deletion_requests
  DROP CONSTRAINT IF EXISTS data_deletion_requests_user_id_fkey;
ALTER TABLE public.data_deletion_requests
  ALTER COLUMN user_id DROP NOT NULL;
ALTER TABLE public.data_deletion_requests
  ADD CONSTRAINT data_deletion_requests_user_id_fkey
  FOREIGN KEY (user_id) REFERENCES auth.users (id) ON DELETE SET NULL;

-- CASCADE delete of consent would fail if the old immutability DELETE
-- trigger were still attached. Keep UPDATE immutability only.
DROP TRIGGER IF EXISTS no_consent_delete ON public.consent_records;
DROP TRIGGER IF EXISTS no_consent_update ON public.consent_records;
CREATE TRIGGER no_consent_update
  BEFORE UPDATE ON public.consent_records
  FOR EACH ROW EXECUTE FUNCTION prevent_consent_mutation();

COMMIT;
