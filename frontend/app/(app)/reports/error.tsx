"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, ArrowLeft, RotateCw } from "lucide-react";

export default function ReportsError({ reset }: { error: Error; reset: () => void }) {
  return (
    <div className="mx-auto max-w-lg">
      <Card>
        <CardHeader className="text-center">
          <div className="mb-2 flex justify-center">
            <div className="rounded-full bg-destructive/10 p-3">
              <AlertTriangle className="h-7 w-7 text-destructive" />
            </div>
          </div>
          <CardTitle>Couldn&apos;t load reports</CardTitle>
          <CardDescription>Please try again.</CardDescription>
        </CardHeader>
        <CardContent className="flex justify-center gap-2">
          <Button variant="outline" onClick={() => reset()}><RotateCw /> Retry</Button>
          <Button asChild><Link href="/home"><ArrowLeft /> Home</Link></Button>
        </CardContent>
      </Card>
    </div>
  );
}
