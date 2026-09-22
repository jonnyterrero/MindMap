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
