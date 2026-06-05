"use server";

import { createClient } from "@/lib/supabase-server";

export type ConversationSummary = {
  id: string;
  title: string | null;
  updated_at: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  crisis_flagged: boolean;
  created_at: string;
};

/** Create a conversation (optionally tied to a journal/check-in entry). Returns its id. */
export async function createConversation(
  contextEntryId?: string | null,
): Promise<{ error: string } | { id: string }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { data, error } = await supabase
    .from("mindmap_ai_conversations")
    .insert({
      user_id: user.id,
      context_entry_id: contextEntryId ?? null,
      title: contextEntryId ? "About a journal entry" : "Reflection",
    })
    .select("id")
    .single();

  if (error) return { error: error.message };
  return { id: data.id as string };
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return [];

  const { data } = await supabase
    .from("mindmap_ai_conversations")
    .select("id, title, updated_at")
    .eq("user_id", user.id)
    .order("updated_at", { ascending: false })
    .limit(50);

  return (data as ConversationSummary[] | null) ?? [];
}

/** Load a conversation's messages (RLS scopes it to the owner). */
export async function getConversationMessages(
  conversationId: string,
): Promise<{ error: string } | { messages: ChatMessage[] }> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { error: "Not authenticated" };

  const { data: conv } = await supabase
    .from("mindmap_ai_conversations")
    .select("id")
    .eq("id", conversationId)
    .eq("user_id", user.id)
    .maybeSingle();
  if (!conv) return { error: "Conversation not found." };

  const { data } = await supabase
    .from("mindmap_ai_messages")
    .select("id, role, content, crisis_flagged, created_at")
    .eq("conversation_id", conversationId)
    .order("created_at", { ascending: true });

  return { messages: (data as ChatMessage[] | null) ?? [] };
}
