import Anthropic from "@anthropic-ai/sdk";
import { createClient } from "@/lib/supabase-server";
import { detectCrisis, CRISIS_RESOURCES } from "@/lib/crisis-detection";

export const runtime = "nodejs";
export const maxDuration = 60;

const SYSTEM_PROMPT = `You are a warm, supportive journaling companion inside MindMap, a personal wellness app. The user is reflecting on their day, mood, sleep, symptoms, and life.

You are NOT a therapist or doctor. Be empathetic, curious, and grounded.

Rules:
- Listen and reflect back; ask gentle, open questions. Keep replies short (2-5 sentences).
- Never diagnose, label a condition, or give medical/clinical/treatment advice.
- Don't be alarmist or prescriptive. No medication advice.
- If the person expresses crisis or self-harm, gently and briefly encourage reaching out to a trusted person or a crisis line (e.g. 988 in the US) — do not try to counsel them through it.
- It's okay to celebrate wins and normalize hard days.`;

/**
 * Streaming chat with the AI companion. Returns plain-text token stream.
 * Crisis detection runs on the user's message; if flagged, the severity is sent
 * back in the `X-Crisis-Severity` header and event id in
 * `X-Crisis-Event-Id` (and a crisis event is logged).
 */
export async function POST(req: Request) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return new Response("Unauthorized", { status: 401 });

  let body: { conversationId?: string; userMessage?: string };
  try {
    body = await req.json();
  } catch {
    return new Response("Bad request", { status: 400 });
  }
  const conversationId = body.conversationId;
  const userMessage = body.userMessage?.trim();
  if (!conversationId || !userMessage) return new Response("Bad request", { status: 400 });

  // Ownership (RLS-backed).
  const { data: conv } = await supabase
    .from("mindmap_ai_conversations")
    .select("id")
    .eq("id", conversationId)
    .eq("user_id", user.id)
    .maybeSingle();
  if (!conv) return new Response("Not found", { status: 404 });

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return new Response("AI is not configured", { status: 503 });

  // Crisis guardrail.
  const severity = detectCrisis(userMessage);
  let crisisEventId: string | null = null;
  if (severity) {
    const { data } = await supabase
      .from("mindmap_crisis_events")
      .insert({
        user_id: user.id,
        severity,
        trigger_source: "ai_message",
        trigger_content_ref: conversationId,
        resources_shown: CRISIS_RESOURCES.map((r) => r.label),
      })
      .select("id")
      .single();
    crisisEventId = (data?.id as string | undefined) ?? null;
  }

  // Persist the user turn, then load the last 20 for context.
  await supabase.from("mindmap_ai_messages").insert({
    conversation_id: conversationId,
    role: "user",
    content: userMessage,
    crisis_flagged: Boolean(severity),
  });

  const { data: history } = await supabase
    .from("mindmap_ai_messages")
    .select("role, content")
    .eq("conversation_id", conversationId)
    .order("created_at", { ascending: false })
    .limit(20);

  const messages = (history ?? [])
    .reverse()
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({ role: m.role as "user" | "assistant", content: m.content as string }));

  const client = new Anthropic({ apiKey });
  const encoder = new TextEncoder();
  let full = "";

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        const s = client.messages.stream({
          model: "claude-opus-4-8",
          max_tokens: 1024,
          system: [{ type: "text", text: SYSTEM_PROMPT, cache_control: { type: "ephemeral" } }],
          messages,
        });
        for await (const event of s) {
          if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
            full += event.delta.text;
            controller.enqueue(encoder.encode(event.delta.text));
          }
        }
      } catch {
        const msg = "\n\nSorry — I had trouble responding just now. Please try again.";
        full += msg;
        controller.enqueue(encoder.encode(msg));
      } finally {
        await supabase.from("mindmap_ai_messages").insert({
          conversation_id: conversationId,
          role: "assistant",
          content: full || "(no response)",
        });
        await supabase
          .from("mindmap_ai_conversations")
          .update({ updated_at: new Date().toISOString() })
          .eq("id", conversationId);
        controller.close();
      }
    },
  });

  const headers: Record<string, string> = { "Content-Type": "text/plain; charset=utf-8" };
  if (severity) headers["X-Crisis-Severity"] = severity;
  if (crisisEventId) headers["X-Crisis-Event-Id"] = crisisEventId;
  return new Response(stream, { headers });
}
