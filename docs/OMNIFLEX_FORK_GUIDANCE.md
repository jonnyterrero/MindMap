# OmniFlex fork guidance (MindMap remains Jonny-owned)

**Status:** Binding product/legal relationship is the signed IP paperwork. This doc is engineering guidance only.

## Relationship model

| Party | Role |
|-------|------|
| Jonny Terrero | Sole owner of `jonnyterrero/MindMap` and upstream roadmap (ML, journal encryption, HIPAA/SOC 2 for *this* product) |
| OmniFlex LLC | May **fork** the public MIT repo and selectively integrate portions into OmniFlex products (e.g. OmniFlex-App) |

There is **no org transfer** and **no requirement** that Jonny migrate MindMap onto Firebase/GCP.

## License posture (MIT — verified)

- Repo license: **MIT** (Copyright Jonny Terrero)
- MIT already permits fork, modify, and commercial sale of derivatives if the copyright notice is preserved
- Anything *beyond* MIT (extra attribution, mandatory upstream contribution, ML/dataset limits, revenue share) must be written into the IP agreement **before** OmniFlex integrates code
- Relicensing upstream later does not revoke MIT rights to snapshots already forked

## Recommended technical approach for OmniFlex

1. Fork into `OmniFlexFitness/` (private recommended once health UX ships)
2. Extract only needed modules (routines UI, tracking forms, chat entry) into OmniFlex’s stack
3. **Port to Firebase / Angular-or-Flutter** inside the OmniFlex fork — do not expect upstream to switch off Supabase/Vercel
4. Improvements made inside the OmniFlex fork stay with OmniFlex unless explicitly contributed back upstream via PR
5. Keep synthetic data until OmniFlex’s own compliance review clears real mental-health data collection

## Compliance fence

HIPAA/SOC 2 exposure follows the **data**, not the GitHub org. OmniFlex’s derived product needs its own review. Jonny’s NYU dataset / psych-department pilot stays on Jonny’s side of the fence.

## Suggested first UX pulls (product backlog)

- Preloaded routine dropdowns
- Quick-access companion chat entry point
- Shared visual language only if brand agreement covers it (OmniFlex neon noir / cyber pastel ≠ MindMap shadcn defaults)

## Checklist before OmniFlex integrates

- [ ] Mutual NDA + IP grant signed (terms beyond MIT explicit)
- [ ] Lawyer sanity-check if revenue share / dataset terms are involved
- [ ] Fork created; copyright notice retained in derivative files
- [ ] No production ingestion of real user mental-health data until compliance OK
- [ ] Upstream `LAUNCH_READINESS_AUDIT.md` reviewed for stub APIs / missing deletion / journal encryption
