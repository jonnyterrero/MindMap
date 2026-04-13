"use client";

import { useState, useTransition } from "react";
import { updateProfile, requestDataDeletion } from "./actions";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Save, Loader2, Check, Trash2, AlertTriangle, Download } from "lucide-react";

type Profile = Record<string, unknown> | null;

const TIMEZONES = [
  "America/New_York",
  "America/Chicago",
  "America/Denver",
  "America/Los_Angeles",
  "America/Anchorage",
  "Pacific/Honolulu",
  "Europe/London",
  "Europe/Paris",
  "Asia/Tokyo",
  "Australia/Sydney",
  "UTC",
];

export function SettingsForm({ profile }: { profile: Profile }) {
  const [isPending, startTransition] = useTransition();
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [displayName, setDisplayName] = useState(
    (profile?.display_name as string) ?? ""
  );
  const [timezone, setTimezone] = useState(
    (profile?.timezone as string) ?? "America/New_York"
  );

  const [showDelete, setShowDelete] = useState(false);
  const [deleteReason, setDeleteReason] = useState("");
  const [deleteSubmitted, setDeleteSubmitted] = useState(false);

  function handleSave() {
    setSaved(false);
    setError(null);
    startTransition(async () => {
      const result = await updateProfile(displayName, timezone);
      if (result.error) {
        setError(result.error);
      } else {
        setSaved(true);
        setTimeout(() => setSaved(false), 3000);
      }
    });
  }

  function handleDeleteRequest() {
    startTransition(async () => {
      const result = await requestDataDeletion("all", deleteReason);
      if (result.error) {
        setError(result.error);
      } else {
        setDeleteSubmitted(true);
        setShowDelete(false);
      }
    });
  }

  return (
    <div className="space-y-6">
      <Card className="glass-card">
        <CardHeader>
          <CardTitle>Profile</CardTitle>
          <CardDescription>
            {profile?.email as string}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Display Name</Label>
            <Input
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label>Timezone</Label>
            <Select value={timezone} onValueChange={setTimezone}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TIMEZONES.map((tz) => (
                  <SelectItem key={tz} value={tz}>
                    {tz.replace(/_/g, " ")}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-3">
            <Button onClick={handleSave} disabled={isPending}>
              {isPending ? (
                <Loader2 className="animate-spin" />
              ) : saved ? (
                <Check />
              ) : (
                <Save />
              )}
              {saved ? "Saved!" : "Save"}
            </Button>
            {error && <p className="text-sm text-destructive">{error}</p>}
          </div>
        </CardContent>
      </Card>

      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Download className="h-5 w-5 text-primary" /> Export My Data
          </CardTitle>
          <CardDescription>
            Download all your MindMap data in JSON or CSV format.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex gap-2">
          <Button variant="outline" asChild>
            <a href="/api/export?format=json" download>
              <Download className="h-4 w-4" /> Export JSON
            </a>
          </Button>
          <Button variant="outline" asChild>
            <a href="/api/export?format=csv" download>
              <Download className="h-4 w-4" /> Export CSV
            </a>
          </Button>
        </CardContent>
      </Card>

      <Card className="glass-card border-destructive/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="h-5 w-5" /> Danger Zone
          </CardTitle>
          <CardDescription>
            Request deletion of your data. This cannot be undone.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {deleteSubmitted ? (
            <p className="text-sm text-green-600">
              Deletion request submitted. We&apos;ll process it and notify you.
            </p>
          ) : !showDelete ? (
            <Button
              variant="destructive"
              onClick={() => setShowDelete(true)}
            >
              <Trash2 className="h-4 w-4" /> Request Data Deletion
            </Button>
          ) : (
            <div className="space-y-3">
              <Textarea
                placeholder="Why are you deleting your data? (optional)"
                value={deleteReason}
                onChange={(e) => setDeleteReason(e.target.value)}
                rows={3}
              />
              <div className="flex gap-2">
                <Button
                  variant="destructive"
                  onClick={handleDeleteRequest}
                  disabled={isPending}
                >
                  {isPending ? <Loader2 className="animate-spin" /> : <Trash2 className="h-4 w-4" />}
                  Confirm Deletion Request
                </Button>
                <Button variant="ghost" onClick={() => setShowDelete(false)}>
                  Cancel
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
