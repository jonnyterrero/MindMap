# MindMap – Mental Health Tracking Application

[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=for-the-badge&logo=vercel)](https://vercel.com/jonnyterreros-projects/v0-mental-health-pwa)

MindMap is a comprehensive mental health Progressive Web App (PWA) built with Next.js and TypeScript. It is a mental health app designed to help and aid individuals with anxiety, ADHD, bipolar, depression, and chronic migraines by giving them an easy way to record, visualize, and reflect on their daily experiences.

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Available Scripts](#available-scripts)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [API Overview](#api-overview)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Mood & Symptom Tracking** – Log mood, anxiety, sleep quality, and physical symptoms with detailed notes.
- **Routine & Medication Management** – Schedule routines, track medication adherence, and set reminders.
- **Analytics Dashboard** – Review trends, correlations, and progress over time with interactive charts.
- **Emergency & Support Tools** – Add crisis contacts, generate quick access plans, and track emergency events.
- **Weather Correlation** – Explore how local weather patterns influence symptoms and overall wellbeing.
- **Offline-Ready PWA** – Install on desktop or mobile and keep using the app with limited connectivity.
- **Accessible UI Library** – Reusable shadcn/ui components power a clean, responsive, and accessible experience.
- **Data Export** – Export personal data in CSV or JSON for sharing or backup.

## Tech Stack

- **Framework:** Next.js 15 (App Router) with TypeScript
- **UI:** Tailwind CSS, shadcn/ui, Radix UI primitives, Lucide icons
- **Charts & Visualization:** Recharts, Embla Carousel
- **State & Forms:** React Hook Form, Zod, custom hooks
- **Database:** Supabase (PostgreSQL) with tracked schema migrations
- **Utilities:** date-fns, class-variance-authority, clsx
- **PWA Support:** Service worker, web manifest, install prompts
- **Analytics:** @vercel/analytics
- **Deployment:** Vercel

## Getting Started

### Prerequisites

- Node.js 18 or higher
- pnpm (recommended), npm, or yarn
- Optional: Python 3.8+ for the legacy FastAPI/Streamlit backend (see `Mindmap.py`)

### Installation

```bash
git clone https://github.com/jonnyterrero/MindMap.git
cd MindMap
pnpm install
```

### Running Locally

```bash
pnpm dev
```

Then open your browser at `http://localhost:3000`.

### Production Build

```bash
pnpm build
pnpm start
```

## Available Scripts

- `pnpm dev` – Start the development server with hot reloading.
- `pnpm build` – Create an optimized production build.
- `pnpm start` – Serve the production build locally.
- `pnpm lint` – Run lint checks using Next.js lint configuration.

## Environment Variables

Create a `.env.local` file for any secrets or third-party API keys:

```
# Example variables
NEXT_PUBLIC_API_BASE_URL=https://api.example.com
WEATHER_API_KEY=your_weather_api_key
```

> `.env*` files are already ignored from version control.

## Project Structure

```
MindMap/
├── app/                    # Next.js App Router pages, layouts, and API routes
├── components/             # Reusable UI components (shadcn/ui)
├── hooks/                  # React hooks, e.g. toast and mobile detection
├── lib/                    # Utility helpers
├── public/                 # Static assets, icons, and service worker
├── styles/                 # Global CSS
├── supabase/               # Supabase migration scripts
├── Mindmap.py              # Legacy FastAPI backend (reference)
├── streamlit_*.py          # Streamlit prototypes and utilities
└── tsconfig.json           # TypeScript configuration
```

## API Overview

The Next.js app exposes REST endpoints under `/api/v1` for mood, sleep, medications, analytics, and authentication. See `API_DOCUMENTATION.md` for full request/response details. A legacy FastAPI backend (`Mindmap.py`) is included for reference or hybrid deployments.

Analytics are captured automatically via `@vercel/analytics` from `app/layout.tsx`.

## Deployment

- **Vercel** – Recommended for the PWA. Connect the GitHub repository and enable automatic deployments on push.
- **Netlify / Railway / Render** – Suitable alternatives with static or serverless hosting.
- **Streamlit Cloud** – For the Python UI referenced in `streamlit_mindtrack_complete.py`.

## Contributing

Contributions, bug reports, and feature requests are always welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to your fork and open a pull request referencing any related issues.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Made with love from me 💗
