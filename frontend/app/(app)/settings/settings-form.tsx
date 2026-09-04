"use client";

import { useState, useTransition } from "react";
import { updateProfile, changePassword } from "./actions";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
import { Save, Loader2, Check, Download, KeyRound } from "lucide-react";

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

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [passwordSaved, setPasswordSaved] = useState(false);

  function handleChangePassword() {
    setPasswordError(null);
    setPasswordSaved(false);
    if (newPassword !== confirmPassword) {
      setPasswordError("New passwords do not match.");
      return;
    }
    startTransition(async () => {
      const result = await changePassword(currentPassword, newPassword);
      if (result.error) {
        setPasswordError(result.error);
      } else {
        setPasswordSaved(true);
        setCurrentPassword("");
        setNewPassword("");
        setConfirmPassword("");
        setTimeout(() => setPasswordSaved(false), 4000);
      }
    });
  }

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
            <KeyRound className="h-5 w-5 text-primary" /> Change Password
          </CardTitle>
          <CardDescription>
            Re-verify your current password, then set a new one (at least 8 characters).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="currentPassword">Current password</Label>
            <Input
              id="currentPassword"
              type="password"
              autoComplete="current-password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="newPassword">New password</Label>
            <Input
              id="newPassword"
              type="password"
              autoComplete="new-password"
              minLength={8}
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="confirmPassword">Confirm new password</Label>
            <Input
              id="confirmPassword"
              type="password"
              autoComplete="new-password"
              minLength={8}
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-3">
            <Button
              onClick={handleChangePassword}
              disabled={
                isPending ||
                !currentPassword ||
                !newPassword ||
                !confirmPassword
              }
            >
              {isPending ? (
                <Loader2 className="animate-spin" />
              ) : passwordSaved ? (
                <Check />
              ) : (
                <KeyRound />
              )}
              {passwordSaved ? "Password updated" : "Update password"}
            </Button>
            {passwordError && (
              <p className="text-sm text-destructive">{passwordError}</p>
            )}
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

      {/*
        Account deletion lives in <DataPrivacy /> (settings/page.tsx) rather
        than here. That path requires typing "DELETE" and cascades immediately
        via admin.auth.admin.deleteUser -- it actually removes data. The old
        "Request Data Deletion" card in this component only inserted a row
        into data_deletion_requests that nothing processed, and told users
        "we'll process it and notify you" -- a false claim under GDPR/CCPA.
      */}
    </div>
  );
}
