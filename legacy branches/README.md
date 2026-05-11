# Legacy branch snapshots

This directory (`legacy branches/`) holds **point-in-time exports** of historical Git branches so the work stays browsable on `main` without keeping long-lived branches around.

Each subfolder is a full tree from `git archive` at the commit listed below (no Git metadata inside—only files).

| Folder | Remote ref | Commit | Tip message |
|--------|------------|--------|-------------|
| `restructure-repo/` | `origin/restructure-repo` | `eabeaae572939c2eb047cf7db81787e5df57b403` | Move Mindmap.py to legacy folder |
| `supabase-setup/` | `origin/supabase-setup` | `e7e3a997c973bdf079a609fd6eb3580f989a76e2` | feat: add medications, journal, goals, therapy, insights, settings, consent, export, body sensations |

To refresh a snapshot later (overwrites that folder):

```bash
git fetch origin
git archive origin/<branch-name> | tar -x -C "legacy branches/<folder-name>"
```

Other remote-only history (for example `cursor/find-recent-app-bulk-c09c`) can be archived the same way if you want it preserved here.
