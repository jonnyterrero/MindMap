-- ============================================================
-- Migration 028: stop duplicating voice transcripts (ADR-003)
-- ============================================================
-- A voice note's spoken text is already stored in its linked journal
-- entry (mindmap_voice_notes.entry_id -> mindmap_journal_entries), which
-- is encrypted under ADR-001. The mindmap_voice_notes.transcript column
-- held a redundant PLAINTEXT copy that nothing ever reads back.
--
-- ADR-003 makes the journal entry the sole, already-encrypted home for the
-- transcript. The app (voice-actions.ts) stops writing this column; this
-- migration nulls any values written before that deploy. Production has 0
-- voice notes at migration time, so this is a forward-looking safety net.
--
-- The column is RETAINED for now and dropped in a later migration after a
-- bake period (safe column-removal sequence: stop writing -> null ->
-- verify -> drop), not dropped here.
-- ============================================================

-- Null any existing plaintext transcripts. Idempotent; re-runnable.
UPDATE public.mindmap_voice_notes
   SET transcript = NULL
 WHERE transcript IS NOT NULL;

COMMENT ON COLUMN public.mindmap_voice_notes.transcript IS
  'DEPRECATED (ADR-003, migration 028): no longer written. The journal '
  'entry referenced by entry_id is the sole, encrypted home for the '
  'transcript. Scheduled for removal in a later migration.';
