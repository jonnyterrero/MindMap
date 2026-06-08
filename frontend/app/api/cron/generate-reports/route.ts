import { NextResponse } from "next/server";
import { createServiceClient } from "@/lib/supabase-server";
import { buildReportForUser } from "@/lib/report-core";

// Scheduled weekly-report generation (Vercel Cron — see frontend/vercel.json).
// Iterates users with recent check-ins and generates a weekly report for each,
// skipping any already generated for the period so re-runs don't re-spend
// tokens. Secured by CRON_SECRET; degrades clearly if env is unconfigured.
export const dynamic = "force-dynamic";
export const maxDuration = 300; // multiple model calls per run

const MAX_USERS_PER_RUN = 200;

export async function GET(request: Request) {
  // Vercel Cron auth: require a matching CRON_SECRET bearer token. Secure by
  // default — if CRON_SECRET is unset, reject (the job stays inert until set).
  const secret = process.env.CRON_SECRET;
  const auth = request.headers.get("authorization");
  if (!secret || auth !== `Bearer ${secret}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    return NextResponse.json({ error: "SUPABASE_SERVICE_ROLE_KEY not configured" }, { status: 503 });
  }
  if (!process.env.ANTHROPIC_API_KEY) {
    return NextResponse.json({ error: "ANTHROPIC_API_KEY not configured" }, { status: 503 });
  }

  const supabase = await createServiceClient();

  // Users who checked in over the last 7 days (service role bypasses RLS).
  const since = new Date();
  since.setDate(since.getDate() - 7);
  const sinceDate = since.toISOString().split("T")[0];

  const { data: rows, error } = await supabase
    .from("mindmap_entries")
    .select("user_id")
    .gte("entry_date", sinceDate);
  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  const userIds = Array.from(
    new Set((rows ?? []).map((r) => r.user_id as string)),
  ).slice(0, MAX_USERS_PER_RUN);

  let generated = 0;
  let skipped = 0;
  let failed = 0;
  for (const userId of userIds) {
    try {
      const res = await buildReportForUser(supabase, userId, "weekly", {
        skipIfExists: true,
      });
      if ("report" in res) generated++;
      else if ("skipped" in res) skipped++;
      else failed++; // e.g. no check-ins inside the report window
    } catch {
      failed++;
    }
  }

  return NextResponse.json({
    ok: true,
    candidates: userIds.length,
    generated,
    skipped,
    failed,
  });
}
