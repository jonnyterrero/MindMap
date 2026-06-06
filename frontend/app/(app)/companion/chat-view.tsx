"use client";

import { useEffect, useRef, useState } from "react";
import { MessageBubble } from "@/components/message-bubble";
import { CrisisResourcesSheet } from "@/components/crisis-resources-sheet";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { ChatMessage } from "./actions";
import type { CrisisSeverity } from "@/lib/crisis-detection";
import { SendHorizontal, Loader2 } from "lucide-react";

type UiMessage = { id: string; role: "user" | "assistant" | "system"; content: string };

export function ChatView({
  conversationId,
  initialMessages,
}: {
  conversationId: string;
  initialMessages: ChatMessage[];
}) {
  const [messages, setMessages] = useState<UiMessage[]>(
    initialMessages.map((m) => ({ id: m.id, role: m.role, content: m.content })),
  );
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [crisis, setCrisis] = useState<CrisisSeverity | null>(null);
  const [crisisEventId, setCrisisEventId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    const text = input.trim();
    if (!text || streaming) return;
    setInput("");
    setStreaming(true);

    const userMsg: UiMessage = { id: `u-${Date.now()}`, role: "user", content: text };
    const assistantId = `a-${Date.now()}`;
    setMessages((m) => [...m, userMsg, { id: assistantId, role: "assistant", content: "" }]);

    try {
      const res = await fetch("/api/ai-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversationId, userMessage: text }),
      });

      const sev = res.headers.get("X-Crisis-Severity") as CrisisSeverity | null;
      const eventId = res.headers.get("X-Crisis-Event-Id");
      if (sev) {
        setCrisis(sev);
        setCrisisEventId(eventId);
      }

      if (!res.ok || !res.body) {
        setMessages((m) =>
          m.map((x) => (x.id === assistantId ? { ...x, content: "Sorry — I couldn't respond. Please try again." } : x)),
        );
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let acc = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        acc += decoder.decode(value, { stream: true });
        setMessages((m) => m.map((x) => (x.id === assistantId ? { ...x, content: acc } : x)));
      }
    } catch {
      setMessages((m) =>
        m.map((x) => (x.id === assistantId ? { ...x, content: "Network error — please try again." } : x)),
      );
    } finally {
      setStreaming(false);
    }
  }

  return (
    <div className="flex h-[calc(100dvh-10rem)] flex-col">
      <div className="flex-1 space-y-3 overflow-y-auto pb-4">
        {messages.length === 0 && (
          <div className="rounded-lg border border-dashed p-6 text-center">
            <p className="text-sm text-muted-foreground">
              Share what&apos;s on your mind. I&apos;ll listen and reflect — no judgment.
            </p>
          </div>
        )}
        {messages.map((m) => (
          <MessageBubble key={m.id} role={m.role} content={m.content} />
        ))}
        {streaming && messages[messages.length - 1]?.content === "" && (
          <div className="flex justify-start">
            <div className="rounded-2xl rounded-bl-sm bg-muted px-3.5 py-2">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t pt-3">
        <MedicalDisclaimer variant="compact" className="mb-2" />
        <div className="flex items-end gap-2">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Write a message…"
            rows={1}
            className="max-h-32 min-h-10 resize-none"
            disabled={streaming}
          />
          <Button size="icon" onClick={send} disabled={streaming || !input.trim()}>
            {streaming ? <Loader2 className="animate-spin" /> : <SendHorizontal />}
          </Button>
        </div>
      </div>

      <CrisisResourcesSheet
        severity={crisis}
        eventId={crisisEventId}
        onClose={() => {
          setCrisis(null);
          setCrisisEventId(null);
        }}
      />
    </div>
  );
}
