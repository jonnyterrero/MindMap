"use client";

import Link from "next/link";
import { useState, useTransition } from "react";
import { grantProviderAccess, revokeProviderAccess, type MyGrant } from "@/app/(app)/provider/actions";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Stethoscope, Loader2, Trash2, ArrowRight } from "lucide-react";

export function ProviderSharing({ grants: initial, isProvider }: { grants: MyGrant[]; isProvider: boolean }) {
  const [grants, setGrants] = useState<MyGrant[]>(initial);
  const [code, setCode] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  const active = grants.filter((g) => !g.revoked_at);

  function share() {
    setErr(null);
    setMsg(null);
    startTransition(async () => {
      const r = await grantProviderAccess(code);
      if ("error" in r) {
        setErr(r.error);
      } else {
        setMsg(`Shared with ${r.providerName}.`);
        setCode("");
        // Optimistically reflect; full data refreshes on next load.
        setGrants((g) => [
          { id: `temp-${Date.now()}`, providerName: r.providerName, granted_at: new Date().toISOString(), revoked_at: null },
          ...g,
        ]);
      }
    });
  }

  function revoke(id: string) {
    setGrants((g) => g.map((x) => (x.id === id ? { ...x, revoked_at: new Date().toISOString() } : x)));
    startTransition(async () => {
      await revokeProviderAccess(id);
    });
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <Stethoscope className="h-5 w-5 text-primary" /> Share with a provider
        </CardTitle>
        <CardDescription>
          Give a clinician or coach read-only access to your reports, predictions, and patterns.
          You can revoke any time.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {isProvider && (
          <Button asChild variant="outline" className="w-full sm:w-auto">
            <Link href="/provider">Open provider dashboard <ArrowRight /></Link>
          </Button>
        )}

        <div className="space-y-2">
          <Label htmlFor="provider-code">Provider code</Label>
          <div className="flex gap-2">
            <Input
              id="provider-code"
              placeholder="Paste the code your provider gave you"
              value={code}
              onChange={(e) => setCode(e.target.value)}
              disabled={isPending}
            />
            <Button onClick={share} disabled={isPending || !code.trim()}>
              {isPending ? <Loader2 className="animate-spin" /> : "Share"}
            </Button>
          </div>
          {err && <p className="text-sm text-destructive">{err}</p>}
          {msg && <p className="text-sm text-green-600">{msg}</p>}
        </div>

        {active.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground">Currently shared with</p>
            {active.map((g) => (
              <div key={g.id} className="flex items-center justify-between rounded-lg border p-2.5">
                <span className="text-sm">{g.providerName}</span>
                <Button size="sm" variant="ghost" onClick={() => revoke(g.id)} disabled={isPending}>
                  <Trash2 className="h-4 w-4 text-destructive" /> Revoke
                </Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
