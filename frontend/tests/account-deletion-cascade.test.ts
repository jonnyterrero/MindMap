import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createClient, type SupabaseClient } from "@supabase/supabase-js";

function loadEnvLocal() {
  const filePath = path.join(process.cwd(), ".env.local");
  if (!fs.existsSync(filePath)) return;
  for (const line of fs.readFileSync(filePath, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const separator = trimmed.indexOf("=");
    if (separator === -1) continue;
    const key = trimmed.slice(0, separator).trim();
    const rawValue = trimmed.slice(separator + 1).trim();
    const value = rawValue.replace(/^["']|["']$/g, "");
    if (!(key in process.env)) process.env[key] = value;
  }
}

loadEnvLocal();

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const configured = Boolean(url && serviceKey);

function adminClient(): SupabaseClient {
  return createClient(url!, serviceKey!, {
    auth: { autoRefreshToken: false, persistSession: false },
  });
}

async function forceCleanup(admin: SupabaseClient, userId: string) {
  await admin.from("data_deletion_requests").delete().eq("user_id", userId);
  await admin.from("consent_records").delete().eq("user_id", userId);
  await admin.from("mindmap_insights").delete().eq("user_id", userId);
  await admin.from("mindmap_journal_entries").delete().eq("user_id", userId);
  await admin.from("mindmap_entries").delete().eq("user_id", userId);
  await admin.auth.admin.deleteUser(userId);
}

async function countFor(
  admin: SupabaseClient,
  table: string,
  userId: string,
): Promise<number> {
  const { count, error } = await admin
    .from(table)
    .select("id", { count: "exact", head: true })
    .eq("user_id", userId);
  if (error) throw new Error(`${table}: ${error.message}`);
  return count ?? 0;
}

test(
  "deleteUser cascades PHI rows and keeps an anonymized deletion audit",
  { skip: !configured },
  async (t) => {
    const admin = adminClient();
    const stamp = Date.now();
    const email = `cascade-test-${stamp}@getmindmapplus.app`;

    const created = await admin.auth.admin.createUser({
      email,
      password: `Tmp-${stamp}-Aa1!`,
      email_confirm: true,
    });
    if (created.error?.message.includes("Legacy API keys")) {
      t.skip(
        "Local SUPABASE_SERVICE_ROLE_KEY is a disabled legacy JWT. Update frontend/.env.local with the rotated secret key, then re-run.",
      );
      return;
    }
    const userId = created.data.user?.id;
    assert.ok(userId, created.error?.message ?? "createUser returned no id");

    let requestId: string | undefined;
    try {
      const today = new Date().toISOString().slice(0, 10);

      const entry = await admin.from("mindmap_entries").insert({
        user_id: userId,
        entry_date: today,
      }).select("id").single();
      assert.equal(entry.error, null, entry.error?.message);

      const journal = await admin.from("mindmap_journal_entries").insert({
        user_id: userId,
        entry_date: today,
        content: "cascade-test journal — delete me",
      }).select("id").single();
      assert.equal(journal.error, null, journal.error?.message);

      const insight = await admin.from("mindmap_insights").insert({
        user_id: userId,
        insight_type: "custom",
        risk_level: "low",
        summary: "cascade-test insight — delete me",
      }).select("id").single();
      assert.equal(insight.error, null, insight.error?.message);

      const consent = await admin.from("consent_records").insert({
        user_id: userId,
        consent_type: "terms_of_service",
        consent_version: "1.0",
        consent_given: true,
      }).select("id").single();
      assert.equal(consent.error, null, consent.error?.message);

      const deletion = await admin.from("data_deletion_requests").insert({
        user_id: userId,
        scope: "all",
        status: "processing",
        reason: "cascade-test",
        retained_metadata: { deleted_user_id: userId },
      }).select("id").single();
      assert.equal(deletion.error, null, deletion.error?.message);
      requestId = deletion.data?.id;

      const deleted = await admin.auth.admin.deleteUser(userId);
      assert.equal(deleted.error, null, deleted.error?.message);

      const authLookup = await admin.auth.admin.getUserById(userId);
      assert.equal(authLookup.data.user, null);

      assert.equal(await countFor(admin, "mindmap_entries", userId), 0);
      assert.equal(await countFor(admin, "mindmap_journal_entries", userId), 0);
      assert.equal(await countFor(admin, "mindmap_insights", userId), 0);
      assert.equal(await countFor(admin, "consent_records", userId), 0);

      assert.ok(requestId);
      const { data: audit, error: auditErr } = await admin
        .from("data_deletion_requests")
        .select("id, user_id, status, retained_metadata")
        .eq("id", requestId)
        .maybeSingle();
      assert.equal(auditErr, null, auditErr?.message);
      assert.ok(audit, "deletion audit row must survive the user delete");
      assert.equal(audit.user_id, null);
      assert.equal(
        (audit.retained_metadata as { deleted_user_id?: string }).deleted_user_id,
        userId,
      );
    } catch (err) {
      await forceCleanup(admin, userId);
      throw err;
    }
  },
);
