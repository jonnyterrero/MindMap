import Anthropic from "@anthropic-ai/sdk";

/**
 * Weekly/monthly wellness report generation via Claude. Takes an aggregated
 * context (computed server-side from the user's own data) and returns a warm,
 * plain-language markdown summary + key insights. Non-diagnostic.
 */
export const REPORT_MODEL = "claude-opus-4-8";

export interface ReportContext {
  reportType: "weekly" | "monthly";
  periodStart: string;
  periodEnd: string;
  stats: Record<string, unknown>;
  topCorrelations: string[];
  predictionAccuracy: { total: number; accurate: number; inaccurate: number };
}

export interface GeneratedReport {
  summaryMarkdown: string;
  keyInsights: string[];
}

const SYSTEM_PROMPT = `You write warm, plain-language wellness summaries for MindMap, a personal self-tracking app. You are NOT a clinician.

Given a user's aggregated self-tracked data for a period, write an encouraging, honest reflection.

Rules:
- Markdown with short section headers (e.g. "## How your week went", "## Patterns worth noticing", "## Gentle suggestions").
- Reference the actual numbers you're given. Be specific but kind.
- Frame patterns as "possible" associations, never causes or diagnoses.
- No medical/clinical/treatment advice; no medication guidance.
- Celebrate consistency and effort. Keep it under ~400 words.`;

export async function generateReportMarkdown(ctx: ReportContext): Promise<GeneratedReport> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("AI reports are not configured (missing ANTHROPIC_API_KEY).");

  const client = new Anthropic({ apiKey });

  const userContent = [
    `Report type: ${ctx.reportType}`,
    `Period: ${ctx.periodStart} to ${ctx.periodEnd}`,
    `Aggregated stats: ${JSON.stringify(ctx.stats)}`,
    `Top possible patterns: ${ctx.topCorrelations.length ? ctx.topCorrelations.join("; ") : "none yet"}`,
    `Prediction accuracy: ${ctx.predictionAccuracy.accurate}/${ctx.predictionAccuracy.total} marked accurate`,
  ].join("\n");

  const response = await client.messages.create({
    model: REPORT_MODEL,
    max_tokens: 4096,
    system: [{ type: "text", text: SYSTEM_PROMPT, cache_control: { type: "ephemeral" } }],
    output_config: {
      effort: "medium",
      format: {
        type: "json_schema",
        schema: {
          type: "object",
          properties: {
            summary_markdown: { type: "string" },
            key_insights: { type: "array", items: { type: "string" } },
          },
          required: ["summary_markdown", "key_insights"],
          additionalProperties: false,
        },
      },
    },
    messages: [{ role: "user", content: userContent }],
  });

  const textBlock = response.content.find(
    (b): b is Anthropic.TextBlock => b.type === "text",
  );
  const parsed = JSON.parse(textBlock?.text ?? "{}");

  return {
    summaryMarkdown: String(parsed.summary_markdown ?? "").trim(),
    keyInsights: Array.isArray(parsed.key_insights) ? parsed.key_insights.map(String).slice(0, 8) : [],
  };
}
