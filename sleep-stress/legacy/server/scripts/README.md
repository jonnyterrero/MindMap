# Python Integration Scripts

This directory contains scripts to integrate the Sleep & Stress + app with your Python analysis system.

## python_integration.py

A comprehensive integration script that connects your Python analysis with the Node.js backend API.

### Features

- **Data Logging**: Log entries that sync to both local file and API
- **Data Sync**: Bidirectional sync between local Python data and API
- **Correlation Analysis**: Run your existing correlation analysis
- **Data Export/Import**: Backup and restore data

### Usage

#### Log a new entry
```bash
python python_integration.py --action log \
  --date "2025-09-26" \
  --sleep-start "23:30" \
  --sleep-end "07:15" \
  --quality 8 \
  --mood 6 \
  --stress 4 \
  --journal "Felt anxious but slept okay." \
  --gi-flare 2 \
  --migraine 1
```

#### Sync data from API to local
```bash
python python_integration.py --action sync-from-api
```

#### Sync local data to API
```bash
python python_integration.py --action sync-to-api
```

#### Run correlation analysis
```bash
python python_integration.py --action analyze
```

#### Export data
```bash
python python_integration.py --action export --file backup.json
```

#### Import data
```bash
python python_integration.py --action import --file backup.json
```

### Integration with Your Existing Code

The script is designed to work seamlessly with your existing Python analysis code:

```python
# Your existing code
from python_integration import SleepStressPlusIntegration

integration = SleepStressPlusIntegration()

# Log entry (same as your log_entry function)
entry = integration.log_entry(
    "2025-09-26", "23:30", "07:15", 8, 6, 4,
    journal="Felt anxious but slept okay.", gi_flare=2, migraine=1
)

# Load dataset (same as your load_dataset function)
df_flat = integration.load_dataset()

# Run correlation analysis
correlations = integration.analyze_correlations()
```

### API Endpoints

The script interacts with these API endpoints:

- `GET /api/health/entries` - Get all entries
- `POST /api/health/entries` - Create new entry
- `GET /api/health/correlation` - Get correlation data
- `GET /api/health/export` - Export all data
- `POST /api/health/import` - Import data

### Configuration

Set the API URL if not using localhost:

```bash
python python_integration.py --api-url "https://your-api.com/api" --action analyze
```

### Error Handling

The script includes comprehensive error handling:
- Network connectivity issues
- API authentication errors
- Data validation errors
- File system errors

### Data Format

The script maintains compatibility with your existing data format:

```json
{
  "date": "2025-09-26",
  "sleep": {
    "start_time": "23:30",
    "end_time": "07:15",
    "duration_hours": 7.75,
    "quality_score": 8
  },
  "mood": {
    "mood_score": 6,
    "stress_score": 4,
    "journal_entry": "Felt anxious but slept okay.",
    "voice_note_path": null
  },
  "symptoms": {
    "gi_flare": 2,
    "skin_flare": 0,
    "migraine": 1
  }
}
```

This ensures seamless integration with your existing Python analysis workflow.
