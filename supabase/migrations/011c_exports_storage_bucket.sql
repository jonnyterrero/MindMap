-- ============================================================
-- Migration 011c: mindmap-exports storage bucket + owner-scoped policies
-- ============================================================
-- Backfill of a migration applied to production on 2026-05-13 (Supabase
-- version 20260513211059, name `exports_storage_bucket`) but never
-- committed. Recovered verbatim on 2026-08-06 via schema_migrations.
--
-- Provisions the private `mindmap-exports` storage bucket the data-export
-- flow writes into, plus RLS on `storage.objects` so a user can read only
-- files under their own UUID prefix. Extends mindmap_data_exports with
-- the columns the exporter uses to track the job lifecycle.
--
-- All idempotent: INSERT ... ON CONFLICT DO NOTHING, DROP POLICY IF
-- EXISTS + CREATE, ALTER TABLE ADD COLUMN IF NOT EXISTS.
-- ============================================================

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'mindmap-exports',
  'mindmap-exports',
  false,
  104857600,
  ARRAY['application/json', 'text/csv', 'application/zip', 'application/pdf']
)
ON CONFLICT (id) DO NOTHING;

DROP POLICY IF EXISTS "mindmap_exports_owner_read" ON storage.objects;
DROP POLICY IF EXISTS "mindmap_exports_service_write" ON storage.objects;

CREATE POLICY "mindmap_exports_owner_read"
ON storage.objects FOR SELECT
TO authenticated
USING (
  bucket_id = 'mindmap-exports'
  AND (storage.foldername(name))[1] = auth.uid()::text
);

CREATE POLICY "mindmap_exports_service_write"
ON storage.objects FOR ALL
TO service_role
USING (bucket_id = 'mindmap-exports')
WITH CHECK (bucket_id = 'mindmap-exports');

ALTER TABLE public.mindmap_data_exports
  ADD COLUMN IF NOT EXISTS storage_path text,
  ADD COLUMN IF NOT EXISTS error_message text,
  ADD COLUMN IF NOT EXISTS requested_at timestamptz NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS started_at timestamptz,
  ADD COLUMN IF NOT EXISTS completed_at timestamptz;
