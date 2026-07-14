# Sleep & Stress Tracker

A comprehensive health tracking application built with Next.js that monitors sleep quality, mood, stress levels, and symptoms with advanced ML and AI integration capabilities.

## 🌟 Overview

Sleep & Stress Tracker is a full-stack web application designed to help users understand the relationships between their sleep patterns, emotional well-being, and physical symptoms. The app combines intuitive data entry, powerful analytics, and machine learning insights to provide personalized health recommendations.

## ✨ Key Features

### Core Health Tracking

#### Sleep Tracking
- **Duration Monitoring**: Track sleep start and end times with automatic duration calculation
- **Quality Scoring**: Rate sleep quality on a 1-10 scale with descriptive feedback
- **Sleep Analytics**: 
  - Average sleep duration and quality trends
  - Consistency analysis and bedtime drift detection
  - Sleep efficiency scoring
  - Weekly, monthly, and yearly trend analysis

#### Mood & Stress Monitoring
- **Mood Scale**: Rate daily mood on a 1-10 scale with emoji feedback
- **Stress Scale**: Track stress levels (1-10) with descriptive levels
- **Journal Entries**: Text-based mood logging for detailed thoughts and experiences
- **Voice Notes**: Audio recording capability for mood tracking (ready for sentiment analysis)
- **Mood Tips**: Contextual suggestions for mood improvement

#### Symptom Tracking
- **GI Flare Monitoring**: Track gastrointestinal symptoms (0-10 scale)
- **Skin Flare Tracking**: Monitor skin condition symptoms (0-10 scale)
- **Migraine Logging**: Record headache and migraine severity (0-10 scale)
- **Symptom Patterns**: Identify trends and correlations with other health metrics

### Advanced Analytics & Insights

#### Correlation Analysis
- **Sleep-Symptom Correlations**: Statistical analysis of relationships between sleep quality and symptoms
- **Mood-Sleep Patterns**: Identify how mood affects sleep quality and vice versa
- **Stress-Health Relationships**: Understand how stress impacts overall well-being
- **Multi-variate Analysis**: Complex relationship mapping between all tracked metrics

#### Personalized Insights
- **Sleep Pattern Analysis**: 
  - Low sleep duration alerts
  - Sleep quality improvement tracking
  - Inconsistent sleep schedule detection
  - Weekend vs. weekday sleep patterns
- **Mood Trend Analysis**:
  - Declining mood alerts
  - Positive mood recognition
  - Mood correlation insights
- **Stress Pattern Detection**:
  - High stress level alerts
  - Rising stress trend warnings
  - Stress management recommendations
- **Lifestyle Factor Analysis**:
  - Weekend sleep catch-up detection
  - Work-life balance insights
  - Routine consistency analysis

#### ML-Powered Predictions
- **Sleep Quality Prediction**: Forecast sleep quality based on mood, stress, and symptoms
- **Mood Prediction**: Predict mood based on sleep patterns and symptoms
- **Symptom Flare Prediction**: Anticipate symptom flares using historical patterns
- **Pattern Recognition**: Advanced algorithms to identify hidden correlations

### User Experience Features

#### Dashboard & Visualizations
- **Comprehensive Dashboard**: Real-time health overview with key metrics
- **Interactive Charts**: 
  - Line charts for trends over time
  - Bar charts for comparisons
  - Correlation heatmaps
  - Sleep analytics visualizations
- **Health Metrics Cards**: Quick access to current status across all tracked areas
- **Quick Log Panel**: Fast entry for daily health data

#### Gamification & Progress
- **Badge System**: Unlock achievements for consistency and improvements
- **Goal Tracking**: Set and monitor personal health goals
- **Progress Visualization**: Track improvement over time
- **Streak Tracking**: Maintain daily logging streaks

#### Profile & Personalization
- **User Profiles**: Store age, gender, BMI, and health information
- **Customizable Goals**: Set personalized targets for sleep, mood, and symptoms
- **Preferences**: Tailor the app experience to individual needs

### AI & Machine Learning Integration

#### Python ML Analytics
The app includes comprehensive Python-based machine learning capabilities:

- **Advanced Sleep Tracker ML** (`ml/ADVANCED_SLEEP_TRACKER_ML.py`):
  - Multi-task learning (regression + classification)
  - Transformer encoder for longer contexts
  - Experiment tracking with Weights & Biases/MLflow
  - SHAP explainability for model insights
  - Data quality checks and outlier clipping
  - Personalized recommendations
  - Coping strategies and mood boosters
  - Voice transcription simulation

- **Complete Sleep Tracker ML** (`ml/COMPLETE_SLEEP_TRACKER_ML.py`):
  - Full-featured ML pipeline
  - Model training and evaluation
  - Prediction generation

- **Fixed Sleep Tracker ML** (`ml/FIXED_SLEEP_TRACKER_ML.py`):
  - Production-ready ML models
  - Optimized for performance

#### ML Features
- **Correlation Analysis**: Statistical correlation calculations between all health metrics
- **Predictive Models**: Linear regression and advanced models for sleep, mood, and symptom prediction
- **Pattern Recognition**: Advanced algorithms to detect sleep consistency and trend patterns
- **Insight Generation**: Automated health recommendations based on data patterns
- **Anomaly Detection**: Identify unusual patterns in health data
- **Time Series Forecasting**: Predict future trends based on historical data

## 🏗️ Technical Architecture

### Frontend Stack
- **Framework**: Next.js 15 with App Router
- **UI Library**: React 19
- **Styling**: Tailwind CSS 4 with custom design system
- **UI Components**: 
  - Radix UI primitives
  - Custom component library
  - Responsive design for all screen sizes
- **Charts & Visualizations**: Recharts, custom chart components
- **State Management**: React hooks and context
- **Type Safety**: TypeScript throughout

### Backend Stack
- **Runtime**: Node.js
- **API**: Next.js API Routes
- **Database**: 
  - SQLite with Drizzle ORM (development)
  - LibSQL client (production-ready)
- **Data Validation**: Zod schemas
- **Authentication**: Better Auth integration

### Server Integration
The project includes a standalone Express.js server (`server_source/`) with:
- RESTful API endpoints
- JSON-based data storage
- Python ML integration
- Analytics services
- Authentication routes
- Health log management

### Machine Learning
- **Language**: Python 3.8+
- **Libraries**: 
  - pandas, numpy for data processing
  - scikit-learn for ML models
  - PyTorch for deep learning
  - SHAP for explainability
  - Matplotlib/Seaborn for visualization
- **Model Types**: 
  - Linear regression
  - Transformer models
  - Multi-task learning models
  - Classification models

### Data Schema

```typescript
{
  date: "2025-09-26",
  sleep: {
    start_time: "23:30",
    end_time: "07:15",
    duration_hours: 7.75,
    quality_score: 8
  },
  mood: {
    mood_score: 6,
    stress_score: 4,
    journal_entry: "Felt anxious but slept okay.",
    voice_note_path: "data/audio/2025-09-26.wav"
  },
  symptoms: {
    gi_flare: 2,      // 0-10 scale
    skin_flare: 0,    // 0-10 scale
    migraine: 1       // 0-10 scale
  }
}
```

## 📁 Project Structure

```
sleepstress/
├── src/
│   ├── app/                    # Next.js app directory
│   │   ├── api/               # API routes
│   │   │   ├── health-logs/   # Health log endpoints
│   │   │   ├── user-badges/   # Badge system endpoints
│   │   │   ├── user-goals/    # Goal tracking endpoints
│   │   │   └── user-profiles/ # User profile endpoints
│   │   ├── page.tsx           # Main dashboard page
│   │   └── layout.tsx         # Root layout
│   ├── components/            # React components
│   │   ├── ui/               # Reusable UI components
│   │   ├── DashboardHeader.tsx
│   │   ├── DashboardMetrics.tsx
│   │   ├── QuickLogPanel.tsx
│   │   ├── SleepTrackingSection.tsx
│   │   ├── MoodStressSection.tsx
│   │   ├── HealthCorrelations.tsx
│   │   ├── SymptomTracking.tsx
│   │   ├── AdvancedInsightsPanel.tsx
│   │   ├── AIInsightsPanel.tsx
│   │   ├── MLInsightsPanel.tsx
│   │   └── ProfileTab.tsx
│   ├── db/                    # Database setup
│   │   ├── schema.ts          # Drizzle schema definitions
│   │   ├── index.ts           # Database client
│   │   └── seeds/             # Seed data
│   ├── lib/                   # Utility libraries
│   │   ├── services/
│   │   │   ├── insights.ts    # Insights generation
│   │   │   └── gamification.ts # Badges and goals
│   │   ├── ml-models.ts       # ML model integration
│   │   └── utils.ts           # Helper functions
│   └── hooks/                 # Custom React hooks
├── ml/                        # Python ML scripts
│   ├── ADVANCED_SLEEP_TRACKER_ML.py
│   ├── COMPLETE_SLEEP_TRACKER_ML.py
│   └── FIXED_SLEEP_TRACKER_ML.py
├── server_source/             # Express.js server (optional)
│   ├── server.js
│   ├── routes/
│   ├── services/
│   └── utils/
├── docs/                      # Documentation
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── SAMPLE_DATA_ANALYSIS.md
│   └── PYCHARM_SETUP_CODE.txt
├── drizzle/                   # Database migrations
├── public/                    # Static assets
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- **Node.js** 20+ 
- **npm**, **yarn**, **pnpm**, or **bun**
- **Python** 3.8+ (for ML features)
- **SQLite** (included with Node.js)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jonnyterrero/sleepstress.git
   cd sleepstress
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   bun install
   ```

3. **Set up Python environment** (for ML features)
   ```bash
   pip install pandas numpy scikit-learn torch matplotlib seaborn shap
   ```

4. **Set up the database**
   ```bash
   npm run db:push
   # or use drizzle-kit to generate migrations
   ```

5. **Run database seeds** (optional)
   ```bash
   npm run db:seed
   ```

### Development

1. **Start the development server**
   ```bash
   npm run dev
   # or
   bun dev
   ```

2. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

### Available Scripts

- `npm run dev` - Start Next.js development server with Turbopack
- `npm run build` - Build the application for production
- `npm run start` - Start the production server
- `npm run lint` - Run ESLint
- `npm run db:push` - Push database schema changes
- `npm run db:studio` - Open Drizzle Studio for database management

### Running the ML Scripts

1. **Prepare your data**
   Export health logs to JSON format compatible with the ML scripts

2. **Run ML analysis**
   ```bash
   python ml/ADVANCED_SLEEP_TRACKER_ML.py
   ```

3. **View results**
   The scripts will generate predictions, correlations, and insights

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file in the root directory:

```env
# Database
DATABASE_URL="file:./local.db"

# Optional: For production database
# DATABASE_URL="libsql://your-database-url"

# API Configuration
API_BASE_URL="http://localhost:3000"

# ML Configuration (if using separate ML server)
ML_API_URL="http://localhost:5000"
```

## 📊 Features in Detail

### Sleep Tracking Features
- Automatic duration calculation from start/end times
- Overnight sleep handling (sleeping past midnight)
- Quality scoring with descriptive feedback
- Sleep factors tracking (caffeine, alcohol, exercise, etc.)
- Sleep notes and additional context
- Consistency scoring and bedtime drift analysis
- Sleep efficiency calculations

### Mood & Stress Features
- Dual-scale rating (mood + stress)
- Journal entry support with text logging
- Voice note recording capability
- Mood improvement tips and suggestions
- Stress management recommendations
- Trend analysis and pattern recognition

### Analytics Features
- Real-time correlation calculations
- Trend analysis over multiple timeframes
- Pattern recognition across all metrics
- Personalized insight generation
- Predictive analytics
- Anomaly detection

### Gamification Features
- Badge system for achievements
- Goal setting and tracking
- Progress visualization
- Streak tracking
- Milestone celebrations

## 🎯 Use Cases

1. **Personal Health Monitoring**: Track daily health metrics to understand personal patterns
2. **Symptom Management**: Identify correlations between sleep, mood, and physical symptoms
3. **Sleep Optimization**: Improve sleep quality through data-driven insights
4. **Stress Management**: Monitor stress levels and receive actionable recommendations
5. **Health Journal**: Maintain a comprehensive health journal with voice and text notes
6. **Medical Consultation Support**: Export data for healthcare provider consultations

## 🔮 Future Enhancements

### Phase 1: Core Features ✅
- [x] Basic sleep and stress tracking
- [x] Mood and symptom logging
- [x] Data persistence and local storage
- [x] User authentication
- [x] Modern UI/UX design
- [x] Backend API with database storage

### Phase 2: ML Integration ✅
- [x] Python backend integration
- [x] Correlation analysis
- [x] Pattern recognition algorithms
- [x] Predictive sleep and stress models
- [ ] Real-time correlation analysis
- [ ] Enhanced ML models

### Phase 3: Advanced Features (In Progress)
- [ ] AI sleep and stress assistant
- [ ] Natural language insights
- [ ] Voice note sentiment analysis
- [ ] Advanced visualizations
- [ ] Data export (CSV/PDF)
- [ ] Report generation

### Phase 4: Integration & Expansion (Planned)
- [ ] Wearable device integration (Fitbit, Apple Watch)
- [ ] Telehealth capabilities
- [ ] Healthcare provider dashboard
- [ ] Community features
- [ ] Mobile app (React Native)
- [ ] Multi-language support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Documentation

Additional documentation is available in the `docs/` directory:
- **IMPLEMENTATION_SUMMARY.md**: Detailed implementation overview
- **SAMPLE_DATA_ANALYSIS.md**: Example data analysis walkthrough
- **PYCHARM_SETUP_CODE.txt**: Development environment setup

## 🛡️ Privacy & Security

- Local data storage with optional cloud sync
- Encrypted sensitive data
- Secure API endpoints
- Input validation and sanitization
- GDPR compliance considerations
- User consent for data collection

## 💡 Tips for Best Results

1. **Consistency**: Log data daily for best insights
2. **Accuracy**: Be honest with ratings for meaningful correlations
3. **Context**: Use journal entries to provide context
4. **Patience**: Allow at least 1-2 weeks of data for meaningful patterns
5. **Review**: Regularly check insights and correlations
6. **Action**: Implement recommendations and track improvements

## 🆘 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

Made with love for Karina P 💜
