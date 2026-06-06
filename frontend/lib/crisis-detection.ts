/**
 * Crisis detection (lightweight keyword tiers)
 * --------------------------------------------
 * NOT a clinical screen. A conservative keyword matcher that surfaces crisis
 * resources when concerning language appears in journals, AI chat, or voice
 * transcripts. Errs toward showing help. Pure & dependency-free.
 */

export type CrisisSeverity = "concern" | "moderate" | "critical";

// Highest tier wins. Phrases are matched case-insensitively as substrings on a
// normalized (collapsed-whitespace) string.
const TIERS: { severity: CrisisSeverity; phrases: string[] }[] = [
  {
    severity: "critical",
    phrases: [
      "suicide",
      "kill myself",
      "killing myself",
      "end my life",
      "end it all",
      "don't want to live",
      "do not want to live",
      "dont want to live",
      "want to die",
      "better off dead",
    ],
  },
  {
    severity: "moderate",
    phrases: ["self-harm", "self harm", "hurt myself", "harm myself", "want to disappear", "cut myself"],
  },
  {
    severity: "concern",
    phrases: ["hopeless", "overwhelmed", "can't cope", "cant cope", "can not cope", "no point", "no reason to go on"],
  },
];

export function detectCrisis(text: string | null | undefined): CrisisSeverity | null {
  if (!text) return null;
  const normalized = text.toLowerCase().replace(/\s+/g, " ");
  for (const tier of TIERS) {
    if (tier.phrases.some((p) => normalized.includes(p))) return tier.severity;
  }
  return null;
}

export interface CrisisResource {
  label: string;
  detail: string;
  href?: string;
}

export const CRISIS_RESOURCES: CrisisResource[] = [
  { label: "988 Suicide & Crisis Lifeline", detail: "Call or text 988 (US, 24/7)", href: "tel:988" },
  { label: "Crisis Text Line", detail: "Text HOME to 741741", href: "sms:741741?&body=HOME" },
  { label: "Emergency services", detail: "Call 911 if you are in immediate danger", href: "tel:911" },
];

export function crisisHeader(severity: CrisisSeverity): { title: string; body: string } {
  switch (severity) {
    case "critical":
      return {
        title: "Your safety matters — help is available now",
        body: "What you wrote sounds really hard. You don't have to face it alone. Please reach out to one of these now.",
      };
    case "moderate":
      return {
        title: "You deserve support",
        body: "It sounds like you're going through something painful. Talking to someone can help.",
      };
    case "concern":
      return {
        title: "You're not alone",
        body: "Tough moments are real. If things feel heavy, support is here whenever you want it.",
      };
  }
}
