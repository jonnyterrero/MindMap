import { NextResponse } from "next/server";
import { createClient, createServiceClient } from "@/lib/supabase-server";
import { isEncryptionEnabled, decryptJournalRows } from "@/lib/journal-crypto";

export async function GET(request: Request) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { searchParams } = new URL(request.url);
  const format = searchParams.get("format") ?? "json";

  const [entries, routines, meds, journal, therapy, goals] = await Promise.all([
    supabase.from("mindmap_entries").select("*").eq("user_id", user.id).order("entry_date", { ascending: false }),
    supabase.from("mindmap_routines").select("*").eq("user_id", user.id),
    supabase.from("mindmap_medication_schedule").select("*").eq("user_id", user.id),
    supabase.from("mindmap_journal_entries").select("*").eq("user_id", user.id).order("entry_date", { ascending: false }),
    supabase.from("mindmap_therapy_sessions").select("*").eq("user_id", user.id).order("session_date", { ascending: false }),
    supabase.from("mindmap_goals").select("*").eq("user_id", user.id),
  ]);

  // GDPR/CCPA: the export must be the user's data in a form they can read.
  // If encryption is on, decrypt journal bodies in-process so the JSON/CSV
  // download contains plaintext, and null out the ciphertext columns so we
  // don't ship both formats side-by-side.
  let journalRows = journal.data ?? [];
  if (isEncryptionEnabled() && journalRows.length > 0) {
    const admin = await createServiceClient();
    const decrypted = await decryptJournalRows(admin, journalRows as Array<Record<string, unknown>>);
    journalRows = decrypted.map((row) => ({
      ...row,
      body_encrypted: null,
      encryption_key_id: null,
      encryption_algo: "none",
      encrypted_at: null,
    }));
  }

  const exportData = {
    exported_at: new Date().toISOString(),
    user_id: user.id,
    entries: entries.data ?? [],
    routines: routines.data ?? [],
    medication_schedules: meds.data ?? [],
    journal_entries: journalRows,
    therapy_sessions: therapy.data ?? [],
    goals: goals.data ?? [],
  };

  await supabase.from("mindmap_data_exports").insert({
    user_id: user.id,
    export_type: format === "csv" ? "CSV" : "JSON",
    export_status: "completed",
  });

  if (format === "csv") {
    const rows = (exportData.entries).map((e: Record<string, unknown>) =>
      Object.values(e).map((v) => JSON.stringify(v ?? "")).join(",")
    );
    const header = exportData.entries.length > 0
      ? Object.keys(exportData.entries[0]).join(",")
      : "";
    const csv = [header, ...rows].join("\n");

    return new NextResponse(csv, {
      headers: {
        "Content-Type": "text/csv",
        "Content-Disposition": `attachment; filename="mindmap-export-${new Date().toISOString().split("T")[0]}.csv"`,
      },
    });
  }

  return new NextResponse(JSON.stringify(exportData, null, 2), {
    headers: {
      "Content-Type": "application/json",
      "Content-Disposition": `attachment; filename="mindmap-export-${new Date().toISOString().split("T")[0]}.json"`,
    },
  });
}
