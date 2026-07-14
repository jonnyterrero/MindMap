# Sleep & Stress Module

This directory packages the Sleep & Stress project for nesting inside other health app repositories.

## Layout

```
sleep-stress/
├── frontend/   # Widget-based Next.js UI rewrite (from local Sleep&Stress)
└── legacy/     # Full original stack (from sleepstress repo clone)
```

## What each section contains

- `frontend/`
  - Modern widget dashboard and cross-app UI.
  - Local storage data model.
  - Intended as the app-facing surface for unified health tracking suites.

- `legacy/`
  - Original Next.js + Drizzle app (`src/`, `drizzle/`).
  - Optional Node/Express API server (`server/` and `server_source/`).
  - Python ML scripts (`ml/`).
  - Historical deployment and setup docs.

## Security note

- `.env*` files are intentionally excluded from this nested module.
- Add environment variables per host repository conventions.

## Basic dev commands

- Frontend rewrite:
  - `cd sleep-stress/frontend`
  - `npm install`
  - `npm run dev`

- Legacy app:
  - `cd sleep-stress/legacy`
  - `npm install`
  - `npm run dev`
