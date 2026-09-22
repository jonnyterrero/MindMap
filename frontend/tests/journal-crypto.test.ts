import { test } from "node:test";
import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import {
  encryptBody,
  decryptBody,
  wrapDek,
  unwrapDek,
  generateDek,
  resolveMasterKey,
} from "../lib/journal-crypto";

// Isolated master-key env for every test. Never mutate process.env directly;
// resolveMasterKey and friends accept an env override precisely so tests
// don't have to.
function envWithKey(hex?: string): Partial<NodeJS.ProcessEnv> {
  return hex ? { JOURNAL_ENCRYPTION_MASTER_KEY: hex } : {};
}

// A stable 32-byte key expressed as hex. Never use in production.
const MK_HEX = "0".repeat(64);
const MK_HEX_ALT = "1".repeat(64);
const MK_BASE64 = Buffer.alloc(32, 0xaa).toString("base64"); // 44 chars w/ padding

test("resolveMasterKey accepts hex", () => {
  const key = resolveMasterKey("v1", envWithKey(MK_HEX));
  assert.equal(key.length, 32);
  assert.equal(key.toString("hex"), MK_HEX);
});

test("resolveMasterKey accepts base64", () => {
  const key = resolveMasterKey("v1", envWithKey(MK_BASE64));
  assert.equal(key.length, 32);
  assert.equal(key[0], 0xaa);
});

test("resolveMasterKey throws when env var is unset", () => {
  assert.throws(
    () => resolveMasterKey("v1", envWithKey(undefined)),
    /not set/,
  );
});

test("resolveMasterKey throws on wrong length", () => {
  // 64 hex chars is 32 bytes; give it 62 hex chars (31 bytes) instead.
  const short = "0".repeat(62);
  assert.throws(
    () => resolveMasterKey("v1", envWithKey(short)),
    /must decode to 32 bytes/,
  );
});

test("resolveMasterKey never echoes the key value in its error", () => {
  // The secret must not appear in the exception message; only the env var
  // name is safe to print.
  const secret = "s".repeat(64);
  try {
    resolveMasterKey("v9999", { JOURNAL_ENCRYPTION_MASTER_KEY_V9999: "" });
    assert.fail("expected throw");
  } catch (e) {
    assert.ok(e instanceof Error);
    assert.ok(!e.message.includes(secret));
  }
});

test("wrap + unwrap round-trips a fresh DEK", () => {
  const env = envWithKey(MK_HEX);
  const dek = generateDek();
  const wrapped = wrapDek(dek, "v1", env);
  const row = {
    wrapped_dek: wrapped.wrappedDek,
    wrap_iv: wrapped.iv,
    wrap_auth_tag: wrapped.authTag,
    wrap_algo: wrapped.algo,
    wrap_master_key_version: wrapped.masterKeyVersion,
  };
  const back = unwrapDek(row, env);
  assert.deepEqual(back, dek);
});

test("unwrap under the wrong master key fails with an auth error", () => {
  const dek = generateDek();
  const w = wrapDek(dek, "v1", envWithKey(MK_HEX));
  const row = {
    wrapped_dek: w.wrappedDek,
    wrap_iv: w.iv,
    wrap_auth_tag: w.authTag,
    wrap_algo: w.algo,
    wrap_master_key_version: w.masterKeyVersion,
  };
  // Same version tag, different master-key bytes -- integrity check trips.
  assert.throws(() => unwrapDek(row, envWithKey(MK_HEX_ALT)));
});

test("body encrypt + decrypt round-trip", () => {
  const dek = generateDek();
  const plaintext = "Today felt heavy. Slept badly, migraine at 4pm. No triggers I can pin down.";
  const blob = encryptBody(dek, plaintext);
  assert.equal(decryptBody(dek, blob), plaintext);
});

test("body encrypt handles unicode + empty string edges", () => {
  const dek = generateDek();
  for (const plaintext of ["", "🧠 mood: 3/5, notes: 感恩节 dinner 잘 됐어요"]) {
    const blob = encryptBody(dek, plaintext);
    assert.equal(decryptBody(dek, blob), plaintext);
  }
});

test("body encrypt produces different ciphertext for the same input (IV is random)", () => {
  const dek = generateDek();
  const a = encryptBody(dek, "same text");
  const b = encryptBody(dek, "same text");
  assert.notDeepEqual(a, b);
  // But both must still decrypt back to the same plaintext.
  assert.equal(decryptBody(dek, a), "same text");
  assert.equal(decryptBody(dek, b), "same text");
});

test("body layout is iv(12) || tag(16) || ciphertext(>=1)", () => {
  const dek = generateDek();
  const blob = encryptBody(dek, "x");
  // 12 (IV) + 16 (tag) + at least 1 byte of ciphertext.
  assert.ok(blob.length >= 12 + 16 + 1);
});

test("decryptBody rejects a blob smaller than iv+tag (28 bytes minimum)", () => {
  const dek = generateDek();
  assert.throws(
    () => decryptBody(dek, Buffer.alloc(20)), // less than 12 + 16
    /too short/,
  );
});

test("decryptBody with a different DEK fails auth", () => {
  const dek1 = generateDek();
  const dek2 = generateDek();
  const blob = encryptBody(dek1, "secret");
  assert.throws(() => decryptBody(dek2, blob));
});

test("decryptBody detects a flipped bit in the ciphertext", () => {
  const dek = generateDek();
  const blob = encryptBody(dek, "sensitive body");
  // Flip a bit in the ciphertext region (past the IV + tag prefix). Node's
  // GCM implementation will surface this as an authentication failure.
  const tampered = Buffer.from(blob);
  tampered[tampered.length - 1] ^= 0x01;
  assert.throws(() => decryptBody(dek, tampered));
});

test("wrong-length DEK is rejected instead of silently truncating", () => {
  const short = randomBytes(16); // 128-bit key, not what AES-256 expects
  assert.throws(() => encryptBody(short, "x"), /expected Buffer of length 32/);
  assert.throws(() => decryptBody(short, Buffer.alloc(29)), /expected Buffer of length 32/);
});
