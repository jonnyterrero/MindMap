import "server-only";
import Anthropic from "@anthropic-ai/sdk";

/**
 * AI journal reflection
 * ---------------------
 * One opt-in Claude call that turns a journal entry into a gentle, NON-clinical
 * reflection: a short summary, one reflective question, and emotional-theme tags.
 * Never diagnoses, never gives medical advice. Server-only — uses ANTHROPIC_API_KEY.
 */

export const REFLECTION_MODEL = "claude-opus-4-8";

export interface JournalReflection {
  summary: string;
  reflectionQuestion: string;
  tags: string[];
}

const SYSTEM_PROMPT = `You are a warm, supportive journaling companion inside MindMap, a personal wellness self-tracking app.

A user shares a journal entry. Respond with a brief, gentle reflection that helps them notice their own patterns. You are NOT a therapist or doctor.

Hard rules:
- Never diagnose, label, or imply a medical or psychiatric condition.
- Never give medical, clinical, or treatment advice, and never suggest medication changes.
- Do not be alarmist. Be kind, grounded, and concise.
- If the entry mentions self-harm or crisis, gently and briefly encourage reaching out to a trusted person or local crisis line — do not attempt to counsel.

Produce exactly:
- summary: 1–2 warm sentences reflecting back what you heard (not advice).
- reflection_question: one open, non-leading question that invites further reflection.
- tags: 3–5 short lowercase emotional-theme tags (e.g. "stress", "gratitude", "sleep", "fatigue", "connection").`;

export async function reflectOnJournalText(content: string): Promise<JournalReflection> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error("AI reflection is not configured (missing ANTHROPIC_API_KEY).");
  }

  const client = new Anthropic({ apiKey });

  const response = await client.messages.create({
    model: REFLECTION_MODEL,
    max_tokens: 1024,
    // Stable system prompt is cached across calls (prompt caching).
    system: [
      { type: "text", text: SYSTEM_PROMPT, cache_control: { type: "ephemeral" } },
    ],
    // Simple, latency-sensitive task — low effort, structured JSON output.
    output_config: {
      effort: "low",
      format: {
        type: "json_schema",
        schema: {
          type: "object",
          properties: {
            summary: { type: "string" },
            reflection_question: { type: "string" },
            tags: { type: "array", items: { type: "string" } },
          },
          required: ["summary", "reflection_question", "tags"],
          additionalProperties: false,
        },
      },
    },
    messages: [
      { role: "user", content: `Here is my journal entry:\n\n${content}` },
    ],
  });

  const textBlock = response.content.find(
    (b): b is Anthropic.TextBlock => b.type === "text",
  );
  const parsed = JSON.parse(textBlock?.text ?? "{}");

  return {
    summary: String(parsed.summary ?? "").trim(),
    reflectionQuestion: String(parsed.reflection_question ?? "").trim(),
    tags: Array.isArray(parsed.tags) ? parsed.tags.map(String).slice(0, 6) : [],
  };
}
