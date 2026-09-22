/**
 * Envelope encryption for journal bodies (ADR-001).
 *
 * Two-layer AES-256-GCM:
 *
 *   master key (MK, from env)   ── wraps ──▶   per-user DEK (data key)
 *                                              ── encrypts ──▶ journal body
 *
 * Only the master key lives in env. DEKs are random 32-byte keys generated
 * once per user, wrapped by MK, and persisted to `mindmap_journal_user_keys`.
 * Journal rows reference the wrapping row by its opaque id via
 * `mindmap_journal_entries.encryption_key_id`.
 *
 * This module exports the pure crypto primitives + envelope wrap/unwrap.
 * The DB-aware "get-or-create user DEK" flow that reads/writes
 * `mindmap_journal_user_keys` is deliberately NOT here yet — that lands
 * with the read/write wire-up in the next slice, along with the app-side
 * `encryptJournalBody` / `decryptJournalBody` wrappers described in ADR-001.
 *
 * Server-only: this file imports `node:crypto` and MUST NOT be reachable
 * from client code. Keep it inside `frontend/lib/` and import only from
 * server-side files (route handlers, server actions, cron jobs, tests).
 */

import { randomBytes, createCipheriv, createDecipheriv } from "node:crypto";

// --- constants -------------------------------------------------------------

const ALGO = "aes-256-gcm" as const;
const KEY_LEN = 32; // 256-bit key
const IV_LEN = 12;  // GCM standard nonce size; 12 bytes gives a full 96-bit IV
const TAG_LEN = 16; // GCM auth tag is always 128 bits

export type BodyAlgo = "aes-256-gcm";

// Everything needed to persist a wrapped DEK to mindmap_journal_user_keys.
export type WrappedDek = {
  wrappedDek: Buffer;
  iv: Buffer;
  authTag: Buffer;
  algo: BodyAlgo;
  masterKeyVersion: string;
};

// What the unwrap step consumes back from the DB row.
export type WrappedDekRow = {
  wrapped_dek: Buffer;
  wrap_iv: Buffer;
  wrap_auth_tag: Buffer;
  wrap_algo: BodyAlgo;
  wrap_master_key_version: string;
};

// --- master key resolution -------------------------------------------------

// Version convention: `JOURNAL_ENCRYPTION_MASTER_KEY` alone is version "v1"
// (the current key). Later rotations add `JOURNAL_ENCRYPTION_MASTER_KEY_V2`
// etc., and the wrap row's `wrap_master_key_version` picks which env var to
// consult when unwrapping. Rotation is out of scope for this slice; the
// version tag exists in every wrap row so a rotation PR doesn't need a
// schema change.
const DEFAULT_MASTER_KEY_VERSION = "v1";

/**
 * Look up a master key by version. Exported for tests; product code should
 * call the higher-level wrap/unwrap functions instead.
 *
 * Accepts either hex (64 chars) or base64 (44 chars incl. `=` padding).
 * The env var is treated as raw text -- do NOT log its value.
 *
 * `env` is typed as Partial<NodeJS.ProcessEnv> rather than NodeJS.ProcessEnv
 * so tests can hand in plain objects without having to fake NODE_ENV.
 * `process.env` (the real caller) still satisfies this shape.
 */
export function resolveMasterKey(
  version: string = DEFAULT_MASTER_KEY_VERSION,
  env: Partial<NodeJS.ProcessEnv> = process.env,
): Buffer {
  const envVar =
    version === "v1"
      ? "JOURNAL_ENCRYPTION_MASTER_KEY"
      : `JOURNAL_ENCRYPTION_MASTER_KEY_${version.toUpperCase()}`;
  const raw = env[envVar];
  if (!raw) {
    // Never quote or log the missing var name's value; the value is a secret.
    throw new Error(
      `Journal encryption is enabled but ${envVar} is not set. Add a 32-byte key (hex or base64) to the environment.`,
    );
  }
  const decoded = decodeKey(raw.trim());
  if (decoded.length !== KEY_LEN) {
    throw new Error(
      `${envVar} must decode to ${KEY_LEN} bytes (got ${decoded.length}). Use a hex-encoded (64 char) or base64-encoded 32-byte value.`,
    );
  }
  return decoded;
}

function decodeKey(raw: string): Buffer {
  // Hex first (64 chars, only [0-9a-f]); fall back to base64 otherwise. A
  // 32-byte value in base64 is 44 chars including padding.
  if (/^[0-9a-fA-F]{64}$/.test(raw)) return Buffer.from(raw, "hex");
  // Buffer.from tolerates padding-optional base64 but returns garbage-length
  // on invalid input; the length check in resolveMasterKey catches that.
  return Buffer.from(raw, "base64");
}

// --- DEK generation --------------------------------------------------------

/**
 * Generate a fresh Data Encryption Key. 32 bytes of CSPRNG entropy.
 */
export function generateDek(): Buffer {
  return randomBytes(KEY_LEN);
}

// --- wrap / unwrap ---------------------------------------------------------

/**
 * Wrap a DEK under the current master key. Result contains everything
 * that needs to persist to mindmap_journal_user_keys.
 */
export function wrapDek(
  dek: Buffer,
  masterKeyVersion: string = DEFAULT_MASTER_KEY_VERSION,
  env: Partial<NodeJS.ProcessEnv> = process.env,
): WrappedDek {
  assertBufferLen(dek, KEY_LEN, "dek");
  const mk = resolveMasterKey(masterKeyVersion, env);
  const iv = randomBytes(IV_LEN);
  const cipher = createCipheriv(ALGO, mk, iv);
  const wrapped = Buffer.concat([cipher.update(dek), cipher.final()]);
  const authTag = cipher.getAuthTag();
  return {
    wrappedDek: wrapped,
    iv,
    authTag,
    algo: ALGO,
    masterKeyVersion,
  };
}

/**
 * Unwrap a DEK from its persisted form. Throws if the auth tag doesn't
 * verify (wrong master key version or ciphertext tampered).
 */
export function unwrapDek(
  row: WrappedDekRow,
  env: Partial<NodeJS.ProcessEnv> = process.env,
): Buffer {
  if (row.wrap_algo !== ALGO) {
    throw new Error(`unwrapDek: unsupported wrap_algo ${row.wrap_algo}`);
  }
  const mk = resolveMasterKey(row.wrap_master_key_version, env);
  const decipher = createDecipheriv(ALGO, mk, row.wrap_iv);
  decipher.setAuthTag(row.wrap_auth_tag);
  const dek = Buffer.concat([
    decipher.update(row.wrapped_dek),
    decipher.final(),
  ]);
  if (dek.length !== KEY_LEN) {
    // Belt-and-braces: GCM auth guarantees integrity, but a stored row with
    // a truncated wrapped_dek would still produce a short DEK on decrypt.
    throw new Error(`unwrapDek: unwrapped DEK has wrong length (${dek.length})`);
  }
  return dek;
}

// --- body encrypt / decrypt ------------------------------------------------

/**
 * Encrypt a UTF-8 plaintext body with the given DEK.
 *
 * Returns a single opaque Buffer laid out as:
 *
 *   iv (12) || auth tag (16) || ciphertext (N)
 *
 * This is what goes into mindmap_journal_entries.body_encrypted (bytea).
 * Packing IV + tag into the body keeps the schema unchanged and avoids
 * partial writes -- either the whole blob is there or nothing is.
 */
export function encryptBody(dek: Buffer, plaintext: string): Buffer {
  assertBufferLen(dek, KEY_LEN, "dek");
  if (plaintext == null) {
    throw new Error("encryptBody: plaintext is required");
  }
  const iv = randomBytes(IV_LEN);
  const cipher = createCipheriv(ALGO, dek, iv);
  const ciphertext = Buffer.concat([
    cipher.update(plaintext, "utf8"),
    cipher.final(),
  ]);
  const tag = cipher.getAuthTag();
  return Buffer.concat([iv, tag, ciphertext]);
}

/**
 * Decrypt a body blob produced by `encryptBody` back to its UTF-8 string.
 * Throws if the tag doesn't verify (wrong DEK or tampering).
 */
export function decryptBody(dek: Buffer, blob: Buffer): string {
  assertBufferLen(dek, KEY_LEN, "dek");
  // An empty plaintext still produces the IV + tag prefix (28 bytes) with
  // zero ciphertext bytes, so the minimum valid blob is IV + TAG. Anything
  // shorter is a truncated / corrupt row.
  if (blob.length < IV_LEN + TAG_LEN) {
    throw new Error(
      `decryptBody: blob too short (${blob.length} bytes); expected at least ${IV_LEN + TAG_LEN} bytes for iv || tag`,
    );
  }
  const iv = blob.subarray(0, IV_LEN);
  const tag = blob.subarray(IV_LEN, IV_LEN + TAG_LEN);
  const ciphertext = blob.subarray(IV_LEN + TAG_LEN);
  const decipher = createDecipheriv(ALGO, dek, iv);
  decipher.setAuthTag(tag);
  return Buffer.concat([decipher.update(ciphertext), decipher.final()]).toString("utf8");
}

// --- small internals -------------------------------------------------------

function assertBufferLen(buf: Buffer, len: number, name: string): void {
  if (!Buffer.isBuffer(buf) || buf.length !== len) {
    throw new Error(`${name}: expected Buffer of length ${len} (got ${buf?.length})`);
  }
}

// --- DB-aware layer --------------------------------------------------------
//
// Encryption is opt-in via the presence of JOURNAL_ENCRYPTION_MASTER_KEY. With
// the master key unset, callers should stay on the legacy plaintext path so
// setting the env var in Vercel is a zero-downtime cutover. With it set:
//   * new writes flow through encryptJournalBodyForUser (creates a wrapped
//     DEK on first use, encrypts the body, returns the DB-shape patch);
//   * reads flow through decryptJournalRow, which handles either format --
//     rows written before the cutover still have plaintext `content` and
//     `encryption_algo = 'none'`, and pass through untouched.
//
// The wrapped-DEK table (mindmap_journal_user_keys) is service-role write
// only, so this file imports the service client lazily. `createClient` (the
// user-session client) is still what queries the journal rows themselves;
// only the key material touches the service role.

// A minimal shape of the Supabase client we accept. Deliberately loose --
// the real client's generics carry a lot of table/row inference that we don't
// need here and can't restate without importing supabase-js just for the
// types. `any` on the returned builders keeps this file cheap and lets both
// the real client and a hand-rolled test fake satisfy the shape.
/* eslint-disable @typescript-eslint/no-explicit-any */
type SupaLike = {
  from: (table: string) => {
    select: (columns: string) => any;
    insert: (row: any) => any;
    update: (patch: any) => any;
  };
};
/* eslint-enable @typescript-eslint/no-explicit-any */

/**
 * True iff journal encryption is enabled in this environment. Callers should
 * branch on this before deciding whether to encrypt on write / attempt
 * decrypt on read. Off by default so a fresh Vercel Preview deploy without
 * the env var configured still works.
 */
export function isEncryptionEnabled(env: Partial<NodeJS.ProcessEnv> = process.env): boolean {
  const raw = env.JOURNAL_ENCRYPTION_MASTER_KEY;
  return typeof raw === "string" && raw.trim().length > 0;
}

/**
 * Read the caller's active wrapped DEK, or create one if none exists yet.
 * Requires a service-role client because mindmap_journal_user_keys is
 * service-role write only.
 *
 * The unique partial index `uq_mindmap_journal_user_keys_active
 * (user_id) WHERE is_active = true` collapses a two-writer race: one of the
 * inserts wins, the other retries the select and finds the winning row.
 */
export async function getOrCreateActiveUserDek(
  admin: SupaLike,
  userId: string,
  env: Partial<NodeJS.ProcessEnv> = process.env,
): Promise<{ keyId: string; dek: Buffer }> {
  const existing = await admin
    .from("mindmap_journal_user_keys")
    .select("id, wrapped_dek, wrap_iv, wrap_auth_tag, wrap_algo, wrap_master_key_version")
    .eq("user_id", userId)
    .eq("is_active", true)
    .maybeSingle();

  if (existing.data) {
    const row = existing.data;
    return {
      keyId: row.id as string,
      dek: unwrapDek(
        {
          wrapped_dek: toBuffer(row.wrapped_dek),
          wrap_iv: toBuffer(row.wrap_iv),
          wrap_auth_tag: toBuffer(row.wrap_auth_tag),
          wrap_algo: row.wrap_algo as BodyAlgo,
          wrap_master_key_version: row.wrap_master_key_version as string,
        },
        env,
      ),
    };
  }

  // No active key yet -- generate + wrap + insert.
  const dek = generateDek();
  const wrapped = wrapDek(dek, undefined, env);
  const keyId = `k_${randomBytes(16).toString("hex")}`;

  const insert = await admin.from("mindmap_journal_user_keys").insert({
    id: keyId,
    user_id: userId,
    wrapped_dek: wrapped.wrappedDek,
    wrap_algo: wrapped.algo,
    wrap_master_key_version: wrapped.masterKeyVersion,
    wrap_iv: wrapped.iv,
    wrap_auth_tag: wrapped.authTag,
    is_active: true,
  });

  if (insert.error) {
    // A concurrent writer won the race -- re-read the winning row.
    const retry = await admin
      .from("mindmap_journal_user_keys")
      .select("id, wrapped_dek, wrap_iv, wrap_auth_tag, wrap_algo, wrap_master_key_version")
      .eq("user_id", userId)
      .eq("is_active", true)
      .maybeSingle();
    if (retry.data) {
      const row = retry.data;
      return {
        keyId: row.id as string,
        dek: unwrapDek(
          {
            wrapped_dek: toBuffer(row.wrapped_dek),
            wrap_iv: toBuffer(row.wrap_iv),
            wrap_auth_tag: toBuffer(row.wrap_auth_tag),
            wrap_algo: row.wrap_algo as BodyAlgo,
            wrap_master_key_version: row.wrap_master_key_version as string,
          },
          env,
        ),
      };
    }
    throw new Error(`getOrCreateActiveUserDek: insert failed: ${insert.error.message}`);
  }

  return { keyId, dek };
}

/**
 * Look up a wrapped DEK by its id (the value stored in
 * `mindmap_journal_entries.encryption_key_id`) and unwrap it. Used on the
 * read path -- a journal row that was encrypted with the previous key still
 * decrypts because rotated keys are kept.
 */
export async function getUserDekById(
  admin: SupaLike,
  keyId: string,
  env: Partial<NodeJS.ProcessEnv> = process.env,
): Promise<Buffer> {
  const { data, error } = await admin
    .from("mindmap_journal_user_keys")
    .select("wrapped_dek, wrap_iv, wrap_auth_tag, wrap_algo, wrap_master_key_version")
    .eq("id", keyId)
    .maybeSingle();
  if (error || !data) {
    throw new Error(`getUserDekById: key ${keyId} not found`);
  }
  return unwrapDek(
    {
      wrapped_dek: toBuffer(data.wrapped_dek),
      wrap_iv: toBuffer(data.wrap_iv),
      wrap_auth_tag: toBuffer(data.wrap_auth_tag),
      wrap_algo: data.wrap_algo as BodyAlgo,
      wrap_master_key_version: data.wrap_master_key_version as string,
    },
    env,
  );
}

/**
 * DB-shape patch that the callers merge into their journal insert/update.
 * The exclusivity CHECK constraint on mindmap_journal_entries (migration 008)
 * enforces `content IS NULL XOR body_encrypted IS NULL` -- content must
 * always be explicitly null when we're writing ciphertext.
 */
export type EncryptedBodyPatch = {
  content: null;
  body_encrypted: Buffer;
  encryption_key_id: string;
  encryption_algo: BodyAlgo;
  encrypted_at: string;
};

/**
 * Encrypt a plaintext body for a user, returning the DB patch the caller
 * spreads into their insert/update. Assumes isEncryptionEnabled(); callers
 * that skip encryption should not call this.
 */
export async function encryptJournalBodyForUser(
  admin: SupaLike,
  userId: string,
  plaintext: string,
  env: Partial<NodeJS.ProcessEnv> = process.env,
): Promise<EncryptedBodyPatch> {
  const { keyId, dek } = await getOrCreateActiveUserDek(admin, userId, env);
  const body_encrypted = encryptBody(dek, plaintext);
  return {
    content: null,
    body_encrypted,
    encryption_key_id: keyId,
    encryption_algo: ALGO,
    encrypted_at: new Date().toISOString(),
  };
}

/**
 * Return a copy of `row` with plaintext `content` populated, decrypting from
 * `body_encrypted` when the row was written under the encrypted path. Rows
 * with `encryption_algo = 'none'` (or missing) pass through untouched, so
 * this is safe to call across a mixed dataset during and after backfill.
 *
 * Callers pass a service-role client so the DEK lookup succeeds without
 * exposing mindmap_journal_user_keys writes to the user session.
 */
export async function decryptJournalRow<
  T extends {
    content?: string | null;
    body_encrypted?: Buffer | Uint8Array | null;
    encryption_algo?: string | null;
    encryption_key_id?: string | null;
  },
>(admin: SupaLike, row: T, env: Partial<NodeJS.ProcessEnv> = process.env): Promise<T> {
  const algo = row.encryption_algo ?? "none";
  if (algo === "none" || row.body_encrypted == null) return row;
  if (algo !== ALGO) {
    throw new Error(`decryptJournalRow: unsupported encryption_algo ${algo}`);
  }
  if (!row.encryption_key_id) {
    throw new Error("decryptJournalRow: encrypted row missing encryption_key_id");
  }
  const dek = await getUserDekById(admin, row.encryption_key_id, env);
  const blob = toBuffer(row.body_encrypted);
  const content = decryptBody(dek, blob);
  return { ...row, content };
}

/**
 * Decrypt multiple rows, deduplicating DEK lookups per key_id. Reads a batch
 * of journal entries in one shot -- typical use: `getJournalEntries()` returns
 * 50 rows, most or all encrypted under the same key, so one unwrap suffices.
 */
export async function decryptJournalRows<
  T extends {
    content?: string | null;
    body_encrypted?: Buffer | Uint8Array | null;
    encryption_algo?: string | null;
    encryption_key_id?: string | null;
  },
>(admin: SupaLike, rows: T[], env: Partial<NodeJS.ProcessEnv> = process.env): Promise<T[]> {
  const cache = new Map<string, Buffer>();
  const out: T[] = new Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const algo = row.encryption_algo ?? "none";
    if (algo === "none" || row.body_encrypted == null) {
      out[i] = row;
      continue;
    }
    if (algo !== ALGO) {
      throw new Error(`decryptJournalRows: unsupported encryption_algo ${algo}`);
    }
    if (!row.encryption_key_id) {
      throw new Error("decryptJournalRows: encrypted row missing encryption_key_id");
    }
    let dek = cache.get(row.encryption_key_id);
    if (!dek) {
      dek = await getUserDekById(admin, row.encryption_key_id, env);
      cache.set(row.encryption_key_id, dek);
    }
    const blob = toBuffer(row.body_encrypted);
    out[i] = { ...row, content: decryptBody(dek, blob) };
  }
  return out;
}

// bytea columns come back as Node Buffer under supabase-js but the type says
// Uint8Array (or `unknown` when the caller passed `select("*")`). Normalize.
function toBuffer(v: Buffer | Uint8Array | unknown): Buffer {
  if (Buffer.isBuffer(v)) return v;
  if (v instanceof Uint8Array) return Buffer.from(v);
  // Fallback for a JSON-decoded {type:"Buffer",data:[...]} shape (rare).
  if (v && typeof v === "object" && "data" in (v as Record<string, unknown>)) {
    const inner = (v as { data: unknown }).data;
    if (Array.isArray(inner)) return Buffer.from(inner as number[]);
  }
  throw new Error("toBuffer: value is not a Buffer / Uint8Array");
}
