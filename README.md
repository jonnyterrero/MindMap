# MindTrack - Mental Health Tracking Application

A comprehensive mental health tracking Progressive Web App (PWA) built with Next.js and TypeScript, merged from the original MindMap Python/Streamlit application.

[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=for-the-badge&logo=vercel)](https://vercel.com/jonnyterreros-projects/v0-mental-health-pwa)
[![Built with Next.js](https://img.shields.io/badge/Built%20with-Next.js-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)

## Overview

MindTrack is a modern, feature-rich mental health tracking application designed to help individuals manage anxiety, ADHD, bipolar disorder, depression, and chronic migraines. The application combines the best features from both the original MindMap Python/Streamlit version and the Next.js PWA version.

## Features

### Core Tracking Features
- **Mood Tracking**: Track daily mood, anxiety levels, energy, and sleep
- **Sleep Tracking**: Monitor sleep duration, quality, bedtime, and wake times
- **Body Symptom Mapping**: Interactive body map to track physical symptoms and pain locations
- **Medication Management**: Track medications, dosages, frequencies, and adherence
- **Routine Tracking**: Manage and track daily routines (morning, evening, exercise, etc.)

### Advanced Features
- **Emergency Alerts**: Track and manage emergency situations with severity levels
- **Weather Correlation**: Analyze how weather conditions affect symptoms and mood
- **Advanced Analytics**: Comprehensive analytics with trends, correlations, and insights
- **Calendar View**: Visual calendar heatmap for tracking patterns over time
- **Data Export**: Export data in JSON or CSV formats for backup and analysis
- **API Integrations**: RESTful API for integrating with other applications
- **Chat Assistant**: AI-powered chat assistant for mental health support

### Technical Features
- **Progressive Web App (PWA)**: Installable on mobile and desktop devices
- **Offline Support**: Service worker for offline functionality
- **Modern UI**: Beautiful glassmorphism design with dark mode support
- **Responsive Design**: Works seamlessly on mobile, tablet, and desktop

## Tech Stack

### Frontend (Primary)
- **Next.js 15**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework
- **Radix UI**: Accessible component primitives
- **Recharts**: Data visualization
- **Lucide React**: Icon library

### Backend (Alternative/Reference)
- **Python FastAPI**: RESTful API backend (see `Mindmap.py`)
- **Streamlit**: Alternative web interface (see `streamlit_mindtrack_complete.py`)
- **SQLAlchemy**: Database ORM
- **Pandas**: Data analysis

## Getting Started

### Prerequisites
- Node.js 18+ and pnpm (or npm/yarn)
- For Python backend: Python 3.8+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jonnyterrero/MindMap.git
   cd MindMap
   ```

2. **Install dependencies**
   ```bash
   pnpm install
   # or
   npm install
   ```

3. **Run the development server**
   ```bash
   pnpm dev
   # or
   npm run dev
   ```

4. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

### Building for Production

```bash
pnpm build
pnpm start
```

## Alternative: Python/Streamlit Backend

The repository includes Python backend files from the original MindMap project:

### FastAPI Backend
```bash
# Install Python dependencies
pip install -r requirements_python.txt

# Run FastAPI server
uvicorn Mindmap:app --reload
```

### Streamlit Interface
```bash
# Install dependencies
pip install -r requirements_python.txt

# Run Streamlit app
streamlit run streamlit_mindtrack_complete.py
```

## Project Structure

```
MindMap/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   │   ├── v1/           # API v1 endpoints
│   │   └── auth/          # Authentication routes
│   ├── page.tsx          # Main application page
│   └── layout.tsx        # Root layout
├── components/            # React components
│   ├── ui/               # UI components (shadcn/ui)
│   └── install-button.tsx
├── lib/                  # Utility functions
├── public/               # Static assets
│   ├── manifest.json    # PWA manifest
│   └── sw.js            # Service worker
├── Mindmap.py           # FastAPI backend (reference)
├── streamlit_mindtrack_complete.py  # Streamlit interface (reference)
├── requirements_python.txt          # Python dependencies
└── streamlit_config.toml            # Streamlit configuration
```

## API Documentation

The application includes a RESTful API for programmatic access. See [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) for complete API documentation.

### Key Endpoints
- `GET /api/v1/mood` - Retrieve mood entries
- `POST /api/v1/mood` - Create mood entry
- `GET /api/v1/sleep` - Retrieve sleep entries
- `POST /api/v1/sleep` - Create sleep entry
- `GET /api/v1/medications` - Retrieve medications
- `GET /api/v1/analytics` - Get analytics data

## Features Comparison

| Feature | Next.js PWA | Python/Streamlit |
|---------|------------|------------------|
| Modern UI/UX | ✅ | ⚠️ |
| Mobile PWA | ✅ | ❌ |
| Offline Support | ✅ | ❌ |
| API Endpoints | ✅ | ✅ |
| Advanced Analytics | ✅ | ✅ |
| Weather Correlation | ✅ | ✅ |
| Emergency Alerts | ✅ | ✅ |
| Data Export | ✅ | ✅ |
| Deployment | Vercel/Netlify | Streamlit Cloud |

## Deployment

### Vercel (Recommended)
1. Push your code to GitHub
2. Import project in Vercel
3. Deploy automatically

### Other Platforms
- **Netlify**: Similar to Vercel
- **Railway**: For full-stack deployment
- **Streamlit Cloud**: For Python/Streamlit version

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Original MindMap project by [@jonnyterrero](https://github.com/jonnyterrero)
- Built with [Next.js](https://nextjs.org/)
- UI components from [shadcn/ui](https://ui.shadcn.com/)
- Icons from [Lucide](https://lucide.dev/)

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This is a merged version combining features from both the original MindMap Python/Streamlit application and the Next.js PWA version. The Next.js version is the primary application, with Python files included as reference/alternative backend options.
