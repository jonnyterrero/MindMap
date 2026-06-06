import Anthropic from "@anthropic-ai/sdk";

/**
 * Lightweight sentiment + theme analysis for a voice-journal transcript.
 * Server-only (imported by the voice "use server" actions). Non-diagnostic.
 */
export const VOICE_MODEL = "claude-opus-4-8";

export interface VoiceSentiment {
  sentiment: number; // -1 (very negative) .. 1 (very positive)
  themes: string[];
}

const SYSTEM_PROMPT = `You analyze a short personal voice-journal transcript for MindMap, a wellness app. You are not a clinician.

Return:
- sentiment: a number from -1 (very negative) to 1 (very positive).
- themes: 3-5 short lowercase emotional-theme tags (e.g. "stress", "gratitude", "fatigue").

Never diagnose or give medical advice.`;

export async function analyzeVoiceTranscript(text: string): Promise<VoiceSentiment> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("Voice analysis is not configured (missing ANTHROPIC_API_KEY).");

  const client = new Anthropic({ apiKey });
  const response = await client.messages.create({
    model: VOICE_MODEL,
    max_tokens: 256,
    system: [{ type: "text", text: SYSTEM_PROMPT, cache_control: { type: "ephemeral" } }],
    output_config: {
      effort: "low",
      format: {
        type: "json_schema",
        schema: {
          type: "object",
          properties: {
            sentiment: { type: "number" },
            themes: { type: "array", items: { type: "string" } },
          },
          required: ["sentiment", "themes"],
          additionalProperties: false,
        },
      },
    },
    messages: [{ role: "user", content: text.slice(0, 4000) }],
  });

  const textBlock = response.content.find((b): b is Anthropic.TextBlock => b.type === "text");
  const parsed = JSON.parse(textBlock?.text ?? "{}");
  const raw = Number(parsed.sentiment);
  return {
    sentiment: Number.isFinite(raw) ? Math.max(-1, Math.min(1, raw)) : 0,
    themes: Array.isArray(parsed.themes) ? parsed.themes.map(String).slice(0, 6) : [],
  };
}
