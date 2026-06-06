"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { saveVoiceNote } from "@/app/(app)/journal/voice-actions";
import { CrisisResourcesSheet } from "@/components/crisis-resources-sheet";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { CrisisSeverity } from "@/lib/crisis-detection";
import { Mic, Square, Loader2, Check } from "lucide-react";

const MAX_SECONDS = 600;

export function VoiceRecorder() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [supported, setSupported] = useState(true);
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [interim, setInterim] = useState("");
  const [seconds, setSeconds] = useState(0);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [crisis, setCrisis] = useState<CrisisSeverity | null>(null);

  // Web Speech API is browser-provided and untyped in lib.dom — use a loose ref.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recRef = useRef<any>(null);
  const finalRef = useRef("");
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  function stop() {
    try {
      recRef.current?.stop();
    } catch {
      /* ignore */
    }
    if (timerRef.current) clearInterval(timerRef.current);
    setRecording(false);
    setInterim("");
  }

  function start() {
    setError(null);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) {
      setSupported(false);
      return;
    }
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-US";
    finalRef.current = transcript ? transcript + " " : "";

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    rec.onresult = (e: any) => {
      let live = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const chunk = e.results[i][0].transcript;
        if (e.results[i].isFinal) finalRef.current += chunk + " ";
        else live += chunk;
      }
      setTranscript(finalRef.current.trim());
      setInterim(live);
    };
    rec.onerror = () => stop();
    rec.onend = () => {
      if (recording) setRecording(false);
    };

    recRef.current = rec;
    rec.start();
    setRecording(true);
    setSeconds(0);
    timerRef.current = setInterval(() => {
      setSeconds((s) => {
        if (s + 1 >= MAX_SECONDS) stop();
        return s + 1;
      });
    }, 1000);
  }

  function submit() {
    const text = transcript.trim();
    if (!text) return;
    setSaving(true);
    setError(null);
    saveVoiceNote(text, seconds).then((r) => {
      setSaving(false);
      if ("error" in r) {
        setError(r.error);
        return;
      }
      if (r.crisis) setCrisis(r.crisis.severity);
      setOpen(false);
      resetState();
      router.refresh();
    });
  }

  function resetState() {
    setTranscript("");
    setInterim("");
    setSeconds(0);
    finalRef.current = "";
  }

  const mmss = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;

  return (
    <>
      <Dialog open={open} onOpenChange={(o) => { if (recording) stop(); setOpen(o); if (!o) resetState(); }}>
        <DialogTrigger asChild>
          <Button variant="outline">
            <Mic className="h-4 w-4" /> Voice note
          </Button>
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Voice note</DialogTitle>
            <DialogDescription>Speak freely — it transcribes on your device.</DialogDescription>
          </DialogHeader>

          {!supported ? (
            <p className="text-sm text-muted-foreground">
              Voice input isn&apos;t supported on this browser. Try Chrome or Safari, or type your entry instead.
            </p>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center justify-center gap-3 py-2">
                {recording ? (
                  <Button variant="destructive" size="lg" onClick={stop}>
                    <Square className="h-4 w-4" /> Stop · {mmss}
                  </Button>
                ) : (
                  <Button size="lg" onClick={start}>
                    <Mic className="h-4 w-4" /> {transcript ? "Resume" : "Start recording"}
                  </Button>
                )}
                {recording && <span className="h-3 w-3 animate-pulse rounded-full bg-red-500" />}
              </div>

              <Textarea
                value={transcript + (interim ? ` ${interim}` : "")}
                onChange={(e) => setTranscript(e.target.value)}
                placeholder="Your words will appear here…"
                rows={5}
                readOnly={recording}
              />

              {error && <p className="text-sm text-destructive">{error}</p>}

              <Button onClick={submit} disabled={saving || recording || !transcript.trim()} className="w-full">
                {saving ? <Loader2 className="animate-spin" /> : <Check />} Save as journal entry
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>

      <CrisisResourcesSheet severity={crisis} onClose={() => setCrisis(null)} />
    </>
  );
}
