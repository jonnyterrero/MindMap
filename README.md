# MindMap – Mental Health Tracking Application

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=flat&logo=vercel)](https://vercel.com/jonnyterreros-projects/v0-mental-health-pwa)

MindMap is a mental health Progressive Web App (PWA) for people managing anxiety, ADHD, bipolar, depression, and chronic migraines. It helps users record, visualize, and reflect on daily experiences.

**Owner:** Jonny Terrero (`jonnyterrero/MindMap`). OmniFlex may **fork** this repo under MIT + any signed IP agreement — there is **no ownership transfer**. See [docs/OMNIFLEX_FORK_GUIDANCE.md](docs/OMNIFLEX_FORK_GUIDANCE.md).

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Repository layout](#repository-layout)
- [Getting Started](#getting-started)
- [Integration-ready modules](#integration-ready-modules)
- [Compliance](#compliance)
- [License](#license)

## Features

- **Mood & Symptom Tracking** – Log mood, anxiety, sleep quality, and physical symptoms with notes
- **Routine & Medication Management** – Schedule routines, track adherence, reminders (wiring still incomplete — see `LAUNCH_READINESS_AUDIT.md`)
- **Analytics Dashboard** – Trends and correlations
- **Emergency & Support Tools** – Crisis contacts and quick-access plans
- **Weather Correlation** – Local weather vs symptoms
- **Offline-Ready PWA** – Installable; limited offline use
- **ML pipeline (research)** – Separate `ml/` package; not production-serving yet

## Tech Stack

| Layer | Choice |
|-------|--------|
| App | Next.js 15 (App Router) + TypeScript in `frontend/` |
| UI | Tailwind CSS, shadcn/ui, Radix, Lucide |
| Data | Supabase (Postgres) + tracked migrations in `supabase/` |
| Mobile shell | Capacitor (`frontend/mobile`, `frontend/capacitor.config.ts`) |
| ML | Python (`ml/` — uv / pyproject) |
| Deploy | Vercel (frontend) |

Legacy FastAPI/Streamlit prototypes are **not** the active path; `backend/` is currently a placeholder.

## Repository layout

```
MindMap/
├── frontend/          # Next.js PWA (canonical app) — run pnpm here
│   ├── app/           # App Router pages + API routes
│   ├── components/    # shadcn/ui + feature UI
│   ├── hooks/
│   ├── lib/
│   ├── e2e/           # Playwright
│   └── mobile/        # Capacitor assets
├── supabase/          # schema.sql + migrations + tests
├── ml/                # Research ML pipeline (not wired to prod yet)
├── backend/           # Placeholder (legacy FastAPI removed from active tree)
├── docs/              # Fork guidance, design notes
├── LAUNCH_READINESS_AUDIT.md
├── DEVELOPMENT_PLAN.md
└── LICENSE            # MIT — Copyright (c) Jonny Terrero
```

## Getting Started

### Prerequisites

- Node.js 18+
- pnpm
- Optional: Python 3.11+ + [uv](https://github.com/astral-sh/uv) for `ml/`

### Install & run (frontend)

```bash
git clone https://github.com/jonnyterrero/MindMap.git
cd MindMap/frontend
pnpm install
cp .env.example .env.local   # fill Supabase + keys
pnpm dev
```

Open `http://localhost:3000`.

```bash
pnpm build && pnpm start   # production locally
pnpm lint
pnpm exec playwright test  # e2e (when configured)
```

### ML package

```bash
cd ml
uv sync
# see ml/README.md and ml/HANDOFF.md
```

## Integration-ready modules

Safer to extract into a product fork (still subject to compliance):

| Module | Location (approx.) | Notes |
|--------|--------------------|-------|
| Routine / medication UI patterns | `frontend/components`, routine routes under `frontend/app` | Good UX candidates for OmniFlex companion features |
| Mood / symptom logging forms | `frontend/app` + form hooks | Port UI; do not reuse fake `/api/v1` stubs as-is |
| Companion chat entry point | product backlog | Suggested UX; verify implementation status before forking |
| Design system primitives | shadcn components under `frontend/components/ui` | MIT-friendly reuse with attribution |

**Do not treat as production-ready without reading `LAUNCH_READINESS_AUDIT.md`:** stub `/api/v1/*` APIs, unwired notifications, plaintext journal columns, incomplete account deletion.

## Compliance

- Mental-health data can trigger **HIPAA / SOC 2** obligations depending on deployment and BAA status
- Upstream MindMap and any OmniFlex **fork** must not ingest real clinical / NYU pilot datasets until each party’s compliance review is done
- Prefer synthetic data for UX work (routine dropdowns, companion chat entry)

## License

MIT — see [LICENSE](LICENSE). Copyright (c) Jonny Terrero.

MIT already allows commercial forks and derivatives with copyright notice retention. Any terms beyond MIT (attribution, upstream contribution, dataset/ML restrictions, revenue share) must be in a separate signed agreement **before** deeper OmniFlex integration.

---

Built for clarity over hype.
