"use client";

import { useState, useTransition } from "react";
import { exportUserData, deleteAccount } from "./data-privacy-actions";
import { updateAnalyticsPreference } from "./actions";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Download, Trash2, Loader2, ShieldCheck } from "lucide-react";
import { ANALYTICS_CONSENT_CHANGED_EVENT } from "@/lib/analytics-consent";

export function DataPrivacy({ analyticsEnabled }: { analyticsEnabled: boolean }) {
  const [isPending, startTransition] = useTransition();
  const [exportErr, setExportErr] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmText, setConfirmText] = useState("");
  const [deleteErr, setDeleteErr] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [analyticsOn, setAnalyticsOn] = useState(analyticsEnabled);
  const [analyticsErr, setAnalyticsErr] = useState<string | null>(null);

  function doExport() {
    setExportErr(null);
    setExporting(true);
    startTransition(async () => {
      const r = await exportUserData();
      setExporting(false);
      if ("error" in r) {
        setExportErr(r.error);
        return;
      }
      const blob = new Blob([JSON.stringify(r.bundle, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `mindmap-export-${new Date().toISOString().split("T")[0]}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });
  }

  function doDelete() {
    setDeleteErr(null);
    setDeleting(true);
    startTransition(async () => {
      const r = await deleteAccount(confirmText.trim());
      setDeleting(false);
      if ("error" in r) {
        setDeleteErr(r.error);
        return;
      }
      window.location.href = "/login";
    });
  }

  function toggleAnalytics(next: boolean) {
    setAnalyticsOn(next);
    setAnalyticsErr(null);
    startTransition(async () => {
      const result = await updateAnalyticsPreference(next);
      if (result.error) {
        setAnalyticsOn(!next);
        setAnalyticsErr(result.error);
        return;
      }

      window.dispatchEvent(
        new CustomEvent(ANALYTICS_CONSENT_CHANGED_EVENT, { detail: next }),
      );
    });
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <ShieldCheck className="h-5 w-5 text-primary" /> Your data &amp; privacy
        </CardTitle>
        <CardDescription>
          Under GDPR/CCPA you can export or permanently delete your data at any time.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-4">
            <div className="space-y-1">
              <Label htmlFor="analytics-toggle">Share de-identified usage analytics</Label>
              <p id="analytics-description" className="text-xs text-muted-foreground">
                Optional. Sends product and performance events with a pseudonymous
                account ID, but no journal text, symptom values, or other personal
                health data.
              </p>
            </div>
            <Switch
              id="analytics-toggle"
              checked={analyticsOn}
              onCheckedChange={toggleAnalytics}
              disabled={isPending}
              aria-describedby="analytics-description"
            />
          </div>
          {analyticsErr && <p className="text-sm text-destructive">{analyticsErr}</p>}
        </div>

        <div className="space-y-2 border-t pt-4">
          <p className="text-sm font-medium">Export my data</p>
          <p className="text-xs text-muted-foreground">Download everything you&apos;ve logged as a JSON file.</p>
          <Button variant="outline" onClick={doExport} disabled={isPending}>
            {exporting ? <Loader2 className="animate-spin" /> : <Download />} Export
          </Button>
          {exportErr && <p className="text-sm text-destructive">{exportErr}</p>}
        </div>

        <div className="space-y-2 border-t pt-4">
          <p className="text-sm font-medium text-destructive">Delete my account</p>
          <p className="text-xs text-muted-foreground">
            Permanently removes your account and all associated data. This cannot be undone.
          </p>

          {!confirmOpen ? (
            <Button variant="outline" className="border-destructive/40 text-destructive" onClick={() => setConfirmOpen(true)}>
              <Trash2 /> Delete account
            </Button>
          ) : (
            <div className="space-y-2 rounded-lg border border-destructive/40 bg-destructive/5 p-3">
              <p className="text-sm">
                Type <span className="font-mono font-semibold">DELETE</span> to confirm. Your data is removed immediately.
              </p>
              <Input
                value={confirmText}
                onChange={(e) => setConfirmText(e.target.value)}
                placeholder="DELETE"
                disabled={deleting}
              />
              <div className="flex gap-2">
                <Button
                  variant="destructive"
                  onClick={doDelete}
                  disabled={deleting || confirmText.trim() !== "DELETE"}
                >
                  {deleting ? <Loader2 className="animate-spin" /> : <Trash2 />} Permanently delete
                </Button>
                <Button variant="ghost" onClick={() => { setConfirmOpen(false); setConfirmText(""); setDeleteErr(null); }} disabled={deleting}>
                  Cancel
                </Button>
              </div>
              {deleteErr && <p className="text-sm text-destructive">{deleteErr}</p>}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
