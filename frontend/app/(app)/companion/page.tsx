import Link from "next/link";
import { listConversations } from "./actions";
import { NewChatButton } from "./new-chat-button";
import { Card, CardContent } from "@/components/ui/card";
import { MessageCircle, ChevronRight } from "lucide-react";

export default async function CompanionPage() {
  const conversations = await listConversations();

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Companion</h1>
          <p className="text-muted-foreground">A gentle space to talk through your day. Not a therapist.</p>
        </div>
        <NewChatButton />
      </div>

      {conversations.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 py-10 text-center">
            <MessageCircle className="h-8 w-8 text-muted-foreground opacity-40" />
            <p className="text-sm text-muted-foreground">No reflections yet. Start a new one above.</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-2">
          {conversations.map((c) => (
            <Link key={c.id} href={`/companion/${c.id}`} className="block">
              <Card className="transition-colors hover:bg-muted/50">
                <CardContent className="flex items-center justify-between gap-3 p-3.5">
                  <div className="flex items-center gap-3">
                    <MessageCircle className="h-4 w-4 text-primary" />
                    <span className="text-sm font-medium">{c.title || "Reflection"}</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    {new Date(c.updated_at).toLocaleDateString()}
                    <ChevronRight className="h-4 w-4" />
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
