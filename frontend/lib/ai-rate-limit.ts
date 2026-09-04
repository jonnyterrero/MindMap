import type { SupabaseClient } from "@supabase/supabase-js";

/**
 * Per-user daily quotas for the four user-initiated AI endpoints.
 * The cron report job is exempt (server-initiated, capped separately).
 * Values are deliberately generous for real use but stop runaway loops
 * and scripted abuse from burning Anthropic spend.
 */
export const AI_DAILY_LIMITS = {
  chat: 150,
  journal_reflection: 25,
  voice_sentiment: 40,
  report: 6,
} as const;

export type AiEndpoint = keyof typeof AI_DAILY_LIMITS;

export type QuotaResult = {
  allowed: boolean;
  used: number;
  quota: number;
};

/**
 * Atomically consume one unit of the caller's daily quota for an endpoint.
 *
 * Fails open on RPC errors (e.g. migration not yet applied in a dev env):
 * the quota protects cost, not data, and the endpoints are already
 * auth-gated — bricking every AI feature on a transient DB error is worse.
 */
export async function consumeAiQuota(
  supabase: SupabaseClient,
  endpoint: AiEndpoint,
): Promise<QuotaResult> {
  const quota = AI_DAILY_LIMITS[endpoint];
  const { data, error } = await supabase.rpc("rpc_consume_ai_quota", {
    p_endpoint: endpoint,
    p_limit: quota,
  });

  if (error) {
    console.error(`[ai-rate-limit] rpc_consume_ai_quota failed for ${endpoint}:`, error.message);
    return { allowed: true, used: 0, quota };
  }

  const row = Array.isArray(data) ? data[0] : data;
  if (!row) return { allowed: true, used: 0, quota };
  return { allowed: Boolean(row.allowed), used: Number(row.used), quota: Number(row.quota) };
}

/** Friendly copy for limit-reached states, shared by the call sites. */
export function quotaExceededMessage(endpoint: AiEndpoint): string {
  switch (endpoint) {
    case "chat":
      return "You've reached today's companion chat limit. It resets at midnight UTC — see you tomorrow.";
    case "journal_reflection":
      return "You've reached today's AI reflection limit. It resets at midnight UTC.";
    case "voice_sentiment":
      return "Daily voice analysis limit reached — your note was still saved.";
    case "report":
      return "You've reached today's report generation limit. It resets at midnight UTC.";
  }
}
