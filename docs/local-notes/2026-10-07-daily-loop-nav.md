# Local note — 2026-10-07 daily loop navigation

Branch: `fix/daily-loop-navigation` (not merged). Written on this machine so the change is visible when the branch lands.

## What changed

Returning sign-in, password reset, consent, and auth callbacks land on `/home`. Finishing onboarding still opens `/today`, because that button is "Start Day 1 Check-In".

Home is the first item in the mobile dock and the desktop nav. Dashboard moved into More. The dock label for settings matches the page title. Home no longer calls Dashboard "History" or Insights "Reports".

The Companion button is hidden on the check-in page and sits above the dock, including the phone home-indicator inset. The Complete check-in control sticks above that dock instead of underneath it.

## Left alone

`docs/adr/001-journal-encryption.md`, `frontend/AGENTS.md`, `frontend/CLAUDE.md`, and the Mac-parked lines in the mobile docs were already dirty and are not part of this branch's commit.
