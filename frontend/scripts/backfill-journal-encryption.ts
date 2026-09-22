#!/usr/bin/env node
/**
 * Backfill plaintext journal bodies to envelope-encrypted rows.
 *
 * ADR-001 wire-up leaves existing rows untouched (they have plaintext
 * `content`, `encryption_algo = 'none'`, `body_encrypted = null`). New writes
 * encrypt from the cutover forward. This script encrypts the historical
 * plaintext rows in place so every row across the table is ciphertext.
 *
 * Prerequisites:
 *   - JOURNAL_ENCRYPTION_MASTER_KEY set in the environment (same value as
 *     Vercel prod).
 *   - SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY set (writes bypass RLS).
 *
 * Usage (dry-run by default):
 *   node --import ./scripts/ts-resolve-hook.mjs scripts/backfill-journal-encryption.ts
 *   node --import ./scripts/ts-resolve-hook.mjs scripts/backfill-journal-encryption.ts --apply
 *
 * Guarantees:
 *   - Idempotent: rows with encryption_algo != 'none' are skipped.
 *   - Batched (default 100 rows/tx-attempt) so a crash mid-run resumes cleanly.
 *   - Never sets both plaintext and ciphertext; the journal_encryption_
 *     exclusivity CHECK would reject that anyway, this script matches it.
 *   - Never logs plaintext bodies.
 */
/* eslint-disable no-console */

import { createClient } from "@supabase/supabase-js";
import { encryptJournalBodyForUser, isEncryptionEnabled } from "../lib/journal-crypto";

const BATCH = 100;

type JournalRow = {
  id: string;
  user_id: string;
  content: string | null;
  encryption_algo: string | null;
};

async function main() {
  const apply = process.argv.includes("--apply");
  const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!url || !serviceKey) {
    console.error("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.");
    process.exit(1);
  }
  if (!isEncryptionEnabled()) {
    console.error("Set JOURNAL_ENCRYPTION_MASTER_KEY (same value as Vercel prod).");
    process.exit(1);
  }

  const admin = createClient(url, serviceKey);
  console.log(apply ? "MODE: apply (writes)" : "MODE: dry-run (no writes; pass --apply to write)");

  let scanned = 0;
  let encrypted = 0;
  let skipped = 0;
  let failed = 0;

  // Page through only the plaintext rows. Once encrypted, they drop out of
  // this filter, so a re-run naturally resumes.
  for (;;) {
    const { data, error } = await admin
      .from("mindmap_journal_entries")
      .select("id, user_id, content, encryption_algo")
      .eq("encryption_algo", "none")
      .not("content", "is", null)
      .order("id", { ascending: true })
      .limit(BATCH);

    if (error) {
      console.error("Select failed:", error.message);
      process.exit(1);
    }
    const rows = (data ?? []) as JournalRow[];
    if (rows.length === 0) break;

    for (const row of rows) {
      scanned++;
      if (row.encryption_algo !== "none" || row.content == null) {
        skipped++;
        continue;
      }
      try {
        const patch = await encryptJournalBodyForUser(admin, row.user_id, row.content);
        if (apply) {
          const { error: upErr } = await admin
            .from("mindmap_journal_entries")
            .update(patch)
            .eq("id", row.id)
            .eq("encryption_algo", "none"); // extra guard against a race
          if (upErr) {
            failed++;
            // Never log the row content.
            console.error(`Update ${row.id} failed: ${upErr.message}`);
            continue;
          }
        }
        encrypted++;
      } catch (e) {
        failed++;
        console.error(
          `Encrypt ${row.id} failed: ${e instanceof Error ? e.message : "unknown"}`,
        );
      }
    }

    // In dry-run mode, the rows still have encryption_algo='none', so the
    // next iteration would loop forever on the same page. Break after one.
    if (!apply) {
      console.log(`Dry-run stopped after first batch of ${rows.length} rows.`);
      break;
    }
    if (rows.length < BATCH) break;
  }

  console.log(
    `Done. scanned=${scanned} encrypted=${encrypted} skipped=${skipped} failed=${failed}`,
  );
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
