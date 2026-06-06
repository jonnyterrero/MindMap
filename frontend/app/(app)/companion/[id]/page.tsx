import Link from "next/link";
import { redirect } from "next/navigation";
import { getConversationMessages } from "../actions";
import { ChatView } from "../chat-view";
import { ArrowLeft } from "lucide-react";

export default async function ConversationPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const res = await getConversationMessages(id);
  if ("error" in res) redirect("/companion");

  return (
    <div className="space-y-2">
      <Link href="/companion" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-3.5 w-3.5" /> Reflections
      </Link>
      <ChatView conversationId={id} initialMessages={res.messages} />
    </div>
  );
}
