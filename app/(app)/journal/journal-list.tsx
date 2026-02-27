"use client";

import { useState, useTransition } from "react";
import { createJournalEntry, deleteJournalEntry, type JournalPayload } from "./actions";
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
import { Plus, Trash2, Loader2, BookOpen, Lock, Globe } from "lucide-react";
import { format, parseISO } from "date-fns";

type Entry = Record<string, unknown>;

const MOOD_TAG_OPTIONS = [
  "happy", "calm", "anxious", "sad", "frustrated",
  "grateful", "hopeful", "tired", "energetic", "overwhelmed",
];

export function JournalList({ entries }: { entries: Entry[] }) {
  const [isPending, startTransition] = useTransition();
  const [showNew, setShowNew] = useState(false);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [moodTags, setMoodTags] = useState<string[]>([]);
  const [isPrivate, setIsPrivate] = useState(true);

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
    startTransition(async () => {
      await createJournalEntry(payload);
      resetForm();
    });
  }

  function handleDelete(id: string) {
    startTransition(async () => {
      await deleteJournalEntry(id);
    });
  }

  return (
    <div className="space-y-4">
      {!showNew ? (
        <Button onClick={() => setShowNew(true)}>
          <Plus className="h-4 w-4" /> New Entry
        </Button>
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
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
