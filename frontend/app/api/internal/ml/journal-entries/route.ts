/**
 * Internal decrypt endpoint for the ML graph pipeline (ADR-002).
 *
 * The graph batch runs in GitHub Actions and must NOT hold the journal master
 * key (a leaked Actions log would then decrypt any stolen DB backup offline,
 * forever). Instead the batch pulls plaintext from here at run time: decryption
 * happens inside the Vercel process that already holds the key, the plaintext
 * travels over TLS, lives only in the runner's memory, and is discarded when
 * the job ends. No second long-lived plaintext copy is created.
 *
 * Auth: a shared bearer secret (ML_JOURNAL_DECRYPT_SECRET), distinct from the
 * master key and instantly rotatable in both Vercel and Actions. Secure by
 * default — if the secret is unset the route rejects, so it stays inert until
 * configured.
 *
 * Node runtime (needs node:crypto via journal-crypto); never cached; never logs
 * decrypted content.
 */

import { NextRequest, NextResponse } from "next/server";
import { timingSafeEqual } from "node:crypto";
import { createServiceClient } from "@/lib/supabase-server";
import { decryptJournalRows } from "@/lib/journal-crypto";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const PAGE_SIZE = 1000; // PostgREST's default row cap; paginate past it.

/** Constant-time bearer check. Returns false when the secret is unconfigured. */
function isAuthorized(req: NextRequest): boolean {
  const secret = process.env.ML_JOURNAL_DECRYPT_SECRET;
  if (!secret) return false;
  const provided = req.headers.get("authorization") ?? "";
  const expected = `Bearer ${secret}`;
  const a = Buffer.from(provided);
  const b = Buffer.from(expected);
  // Length guard first — timingSafeEqual throws on unequal lengths.
  return a.length === b.length && timingSafeEqual(a, b);
}

type JournalRow = {
  id: string;
  user_id: string;
  entry_date: string | null;
  content: string | null;
  body_encrypted: Buffer | Uint8Array | null;
  encryption_algo: string | null;
  encryption_key_id: string | null;
};

export async function GET(req: NextRequest) {
  if (!isAuthorized(req)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  // Optional watermark: only entries updated at/after `since` (ISO 8601). Bounds
  // how much plaintext a single call can surface; the batch's content_sha skip
  // still prevents recomputing unchanged entries, so this never drops work.
  const since = req.nextUrl.searchParams.get("since");

  const admin = await createServiceClient();

  const rows: JournalRow[] = [];
  for (let from = 0; ; from += PAGE_SIZE) {
    let query = admin
      .from("mindmap_journal_entries")
      .select(
        "id, user_id, entry_date, content, body_encrypted, encryption_algo, encryption_key_id",
      )
      .is("deleted_at", null)
      .order("updated_at", { ascending: true })
      .range(from, from + PAGE_SIZE - 1);
    if (since) query = query.gte("updated_at", since);

    const { data, error } = await query;
    if (error) {
      return NextResponse.json({ error: error.message }, { status: 500 });
    }
    if (!data || data.length === 0) break;
    rows.push(...(data as JournalRow[]));
    if (data.length < PAGE_SIZE) break;
  }

  // Decrypt encrypted rows (legacy plaintext rows pass through). Throws loudly
  // if a row is encrypted but the master key is missing — a misconfigured
  // endpoint must fail, not silently drop entries.
  const decrypted = await decryptJournalRows(admin, rows);

  const entries = decrypted
    .filter((r) => typeof r.content === "string" && r.content.trim() !== "")
    .map((r) => ({
      id: r.id,
      user_id: r.user_id,
      entry_date: r.entry_date,
      content: r.content,
    }));

  return NextResponse.json({ count: entries.length, entries });
}
