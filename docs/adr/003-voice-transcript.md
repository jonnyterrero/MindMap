# ADR-003 — Voice transcripts: one encrypted home, not a plaintext duplicate

- **Status:** Accepted (2026-10-07). Implemented on `feat/adr-003-voice-transcript`.
- **Decision makers:** Jonny (product). Recorded so engineering can proceed without re-litigating.
- **Related:** [ADR-001](001-journal-encryption.md) (journal envelope encryption); launch audit follow-up "`mindmap_voice_notes.transcript` still plaintext"; `frontend/app/(app)/journal/voice-actions.ts`.

## Context

A voice note is captured with the browser Web Speech API: transcription happens client-side and **no audio file is stored** (`mindmap_voice_notes.storage_path` is a `webspeech://<ts>` placeholder, not a blob). So the only sensitive artifact is the transcript text.

`saveVoiceNote()` already creates a linked journal entry from that text (`mindmap_voice_notes.entry_id` → `mindmap_journal_entries.id`), and the journal body is encrypted under ADR-001 when encryption is enabled. It **also** writes the same text, in plaintext, into `mindmap_voice_notes.transcript`.

Findings at decision time:

- The transcript column is **write-only**: nothing in the app or ML pipeline ever reads it back. Display, AI reflection, crisis detection and sentiment all run off the journal entry or the in-memory text at write time — never the stored `transcript`.
- It only appears in the data export/delete table list.
- Production `mindmap_voice_notes` currently holds **0 rows**, so this is a forward-looking change with no existing data to migrate or lose.

So this is not "a plaintext field that needs encrypting." It is a redundant plaintext copy of content that already has an encrypted home.

## Options considered

1. **Encrypt the transcript column** (mirror ADR-001 on `voice_notes`). Keeps the duplicate, just encrypted; adds a second encrypt/decrypt + key-rotation + export path to maintain forever.
2. **Eliminate the duplicate.** Stop writing `transcript`; the linked journal entry is the sole, already-encrypted home. Null any existing values.
3. **Stop-writing but keep the column** unused and unencrypted.

## Decision

**Option 2. The linked journal entry is the single source of truth for a voice transcript; `mindmap_voice_notes.transcript` is no longer written and existing values are nulled.**

- `saveVoiceNote()` stops persisting `transcript` (writes `NULL`). It still creates the encrypted journal entry, runs crisis detection and sentiment on the in-memory text, and stores `sentiment_score` / `themes` (derived, low-sensitivity — out of scope here).
- Migration **028** nulls any existing `transcript` values (0 rows today; idempotent) and comments the column as deprecated. The column itself is **retained for now** and dropped in a later migration after a bake period — a standard safe column-removal sequence, not an irreversible schema change in the same step.
- **Load-bearing assumption, written down so a future change re-opens this ADR instead of silently re-adding plaintext:** a voice note is always backed by a journal entry, which is its canonical text. If product later decouples voice notes (a first-class voice-note view, or notes with no journal entry), the fix is to make the voice note *itself* the single **encrypted** home and stop auto-creating the journal entry — never to re-introduce a second copy.

## Consequences

- No plaintext transcript duplicate exists anywhere: nothing to encrypt, rotate, export, or leak. Strongest data-minimization outcome and the least code.
- Removes a silent-divergence bug class (editing the journal entry no longer leaves a stale transcript copy).
- **Behavior change:** a voice note whose journal entry is later deleted (`entry_id` → `NULL`) retains only metadata (`duration_seconds`, `sentiment_score`, `themes`), not the spoken text. This is consistent with the user having deleted that journal entry, and is accepted.
- The deferred column `DROP` is tracked as a follow-up migration.

## Deferred

- Drop `mindmap_voice_notes.transcript` after confirming no regressions in a release.
- Whether `sentiment_score` / `themes` warrant their own minimization pass (separate decision).
