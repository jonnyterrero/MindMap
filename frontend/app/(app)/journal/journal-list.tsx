"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import {
  createJournalEntry,
  deleteJournalEntry,
  reflectOnJournalEntry,
  type JournalPayload,
  type JournalAnalysis,
} from "./actions";
import { createConversation } from "@/app/(app)/companion/actions";
import { VoiceRecorder } from "@/components/voice-recorder";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { MedicalDisclaimer } from "@/components/medical-disclaimer";
import { CrisisResourcesSheet } from "@/components/crisis-resources-sheet";
import type { CrisisSeverity } from "@/lib/crisis-detection";
import { Plus, Trash2, Loader2, BookOpen, Lock, Globe, Sparkles, MessageCircle } from "lucide-react";
import { format, parseISO } from "date-fns";

type Entry = Record<string, unknown>;

const MOOD_TAG_OPTIONS = [
  "happy", "calm", "anxious", "sad", "frustrated",
  "grateful", "hopeful", "tired", "energetic", "overwhelmed",
];

export function JournalList({
  entries: initialEntries,
  aiEnabled = false,
  analyses = [],
}: {
  entries: Entry[];
  aiEnabled?: boolean;
  analyses?: JournalAnalysis[];
}) {
  const [isPending, startTransition] = useTransition();
  const [entries, setEntries] = useState(initialEntries);
  const [showNew, setShowNew] = useState(false);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [moodTags, setMoodTags] = useState<string[]>([]);
  const [isPrivate, setIsPrivate] = useState(true);

  const [analysisMap, setAnalysisMap] = useState<Record<string, JournalAnalysis>>(
    Object.fromEntries(analyses.map((a) => [a.journal_entry_id, a])),
  );
  const [reflectingId, setReflectingId] = useState<string | null>(null);
  const [reflectErrors, setReflectErrors] = useState<Record<string, string>>({});
  const [crisis, setCrisis] = useState<{ severity: CrisisSeverity; eventId: string | null } | null>(null);
  const router = useRouter();
  const [talkingId, setTalkingId] = useState<string | null>(null);

  function talkAboutEntry(entryId: string) {
    setTalkingId(entryId);
    startTransition(async () => {
      const r = await createConversation(entryId);
      if ("id" in r) router.push(`/companion/${r.id}`);
      else setTalkingId(null);
    });
  }

  async function handleReflect(entryId: string) {
    setReflectingId(entryId);
    setReflectErrors((prev) => {
      const next = { ...prev };
      delete next[entryId];
      return next;
    });
    try {
      const res = await reflectOnJournalEntry(entryId);
      if ("error" in res) {
        setReflectErrors((prev) => ({ ...prev, [entryId]: res.error }));
      } else {
        setAnalysisMap((prev) => ({ ...prev, [entryId]: res.analysis }));
      }
    } finally {
      setReflectingId(null);
    }
  }

  function resetForm() {
    setTitle("");
    setContent("");
    setMoodTags([]);
    setIsPrivate(true);
    setShowNew(false);
  }

  function toggleTag(tag: string) {
    setMoodTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  }

  function handleCreate() {
    if (!content.trim()) return;
    const payload: JournalPayload = {
      entry_date: new Date().toISOString().split("T")[0],
      title: title.trim() || null,
      content: content.trim(),
      mood_tags: moodTags,
      is_private: isPrivate,
    };

    const optimistic: Entry = {
      id: `temp-${Date.now()}`,
      ...payload,
      created_at: new Date().toISOString(),
    };
    setEntries((prev) => [optimistic, ...prev]);
    resetForm();

    startTransition(async () => {
      const res = await createJournalEntry(payload);
      if (res && "success" in res && res.crisis) {
        setCrisis({ severity: res.crisis.severity, eventId: res.crisis.eventId });
      }
    });
  }

  function handleDelete(id: string) {
    setEntries((prev) => prev.filter((e) => e.id !== id));
    startTransition(async () => {
      await deleteJournalEntry(id);
    });
  }

  return (
    <div className="space-y-4">
      {!showNew ? (
        <div className="flex gap-2">
          <Button onClick={() => setShowNew(true)}>
            <Plus className="h-4 w-4" /> New Entry
          </Button>
          <VoiceRecorder />
        </div>
      ) : (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-base">New Journal Entry</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Title (optional)</Label>
              <Input
                placeholder="What's on your mind?"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label>Content</Label>
              <Textarea
                placeholder="Write freely..."
                value={content}
                onChange={(e) => setContent(e.target.value)}
                rows={6}
              />
            </div>

            <div className="space-y-2">
              <Label>Mood tags</Label>
              <div className="flex flex-wrap gap-2">
                {MOOD_TAG_OPTIONS.map((tag) => (
                  <button
                    key={tag}
                    type="button"
                    onClick={() => toggleTag(tag)}
                    className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                      moodTags.includes(tag)
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted text-muted-foreground hover:bg-muted/80"
                    }`}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="private" className="flex items-center gap-2">
                {isPrivate ? <Lock className="h-4 w-4" /> : <Globe className="h-4 w-4" />}
                {isPrivate ? "Private" : "Shareable"}
              </Label>
              <Switch
                id="private"
                checked={isPrivate}
                onCheckedChange={setIsPrivate}
              />
            </div>

            <div className="flex gap-2">
              <Button onClick={handleCreate} disabled={isPending || !content.trim()}>
                {isPending ? <Loader2 className="animate-spin" /> : <Plus className="h-4 w-4" />}
                Save
              </Button>
              <Button variant="ghost" onClick={resetForm}>
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {entries.length === 0 && !showNew ? (
        <p className="text-center text-muted-foreground py-8">
          No journal entries yet. Start writing above.
        </p>
      ) : (
        <div className="space-y-3">
          {entries.map((entry) => (
            <Card key={entry.id as string} className="glass-card">
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-base flex items-center gap-2">
                      <BookOpen className="h-4 w-4 text-primary" />
                      {(entry.title as string) || "Untitled"}
                      {(entry.is_private as boolean) && (
                        <Lock className="h-3 w-3 text-muted-foreground" />
                      )}
                    </CardTitle>
                    <CardDescription>
                      {format(parseISO(entry.entry_date as string), "EEEE, MMMM d, yyyy")}
                    </CardDescription>
                  </div>
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={() => handleDelete(entry.id as string)}
                    disabled={isPending}
                  >
                    <Trash2 className="h-4 w-4 text-destructive" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm whitespace-pre-wrap">{entry.content as string}</p>
                {(entry.mood_tags as string[])?.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-3">
                    {(entry.mood_tags as string[]).map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-0.5 rounded-full text-xs bg-primary/10 text-primary"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}

                {!(entry.id as string).startsWith("temp-") && (
                  <button
                    type="button"
                    onClick={() => talkAboutEntry(entry.id as string)}
                    disabled={talkingId === (entry.id as string)}
                    className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
                  >
                    {talkingId === (entry.id as string) ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <MessageCircle className="h-3 w-3" />
                    )}
                    Talk to AI about this entry
                  </button>
                )}

                {aiEnabled && !(entry.id as string).startsWith("temp-") && (
                  <div className="mt-4 border-t pt-3">
                    {analysisMap[entry.id as string] ? (
                      <div className="space-y-2 rounded-lg bg-muted/50 p-3">
                        <p className="flex items-center gap-1.5 text-xs font-medium text-primary">
                          <Sparkles className="h-3.5 w-3.5" /> AI reflection
                        </p>
                        <p className="text-sm">{analysisMap[entry.id as string].summary}</p>
                        {analysisMap[entry.id as string].reflection_question && (
                          <p className="text-sm italic text-muted-foreground">
                            {analysisMap[entry.id as string].reflection_question}
                          </p>
                        )}
                        {analysisMap[entry.id as string].tags?.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {analysisMap[entry.id as string].tags.map((t) => (
                              <span
                                key={t}
                                className="rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary"
                              >
                                {t}
                              </span>
                            ))}
                          </div>
                        )}
                        <MedicalDisclaimer variant="compact" />
                      </div>
                    ) : (
                      <div className="space-y-1">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleReflect(entry.id as string)}
                          disabled={reflectingId === (entry.id as string)}
                        >
                          {reflectingId === (entry.id as string) ? (
                            <Loader2 className="animate-spin" />
                          ) : (
                            <Sparkles className="h-4 w-4" />
                          )}
                          Reflect with AI
                        </Button>
                        {reflectErrors[entry.id as string] && (
                          <p className="text-xs text-destructive">
                            {reflectErrors[entry.id as string]}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <CrisisResourcesSheet
        severity={crisis?.severity ?? null}
        eventId={crisis?.eventId}
        onClose={() => setCrisis(null)}
      />
    </div>
  );
}
