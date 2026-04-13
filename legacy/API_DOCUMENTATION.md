# MindTrack API Documentation

## Overview

The MindTrack API allows you to integrate your mental health tracking data with other applications and services. This RESTful API provides endpoints for accessing mood, sleep, medication, and analytics data.

## Base URL

\`\`\`
https://your-domain.com/api/v1
\`\`\`

## Authentication

All API requests require authentication using an API key. Include your API key in the request header:

\`\`\`bash
curl -H "x-api-key: YOUR_API_KEY" \
  https://your-domain.com/api/v1/mood
\`\`\`

### Generating API Keys

1. Navigate to the Integrations tab in MindTrack
2. Click "Generate New Key"
3. Copy and securely store your API key
4. Use the key in the `x-api-key` header for all requests

## Rate Limits

- 1,000 requests per hour
- 10,000 requests per day

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Time when the rate limit resets

## Endpoints

### Mood Tracking

#### Get Mood Entries

\`\`\`http
GET /api/v1/mood
\`\`\`

Retrieve mood tracking entries with optional filtering.

**Query Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| startDate | string | Filter by start date (YYYY-MM-DD) | - |
| endDate | string | Filter by end date (YYYY-MM-DD) | - |
| limit | integer | Number of results to return | 100 |

**Example Request:**

\`\`\`bash
curl -H "x-api-key: YOUR_API_KEY" \
  "https://your-domain.com/api/v1/mood?startDate=2024-01-01&limit=50"
\`\`\`

**Example Response:**

\`\`\`json
{
  "data": [
    {
      "id": "1",
      "date": "2024-01-15",
      "mood": 8,
      "anxiety": 3,
      "energy": 7,
      "notes": "Feeling great today",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "meta": {
    "total": 1,
    "limit": 100,
    "startDate": "2024-01-01",
    "endDate": null
  }
}
\`\`\`

#### Create Mood Entry

\`\`\`http
POST /api/v1/mood
\`\`\`

Create a new mood tracking entry.

**Request Body:**

\`\`\`json
{
  "mood": 8,
  "anxiety": 3,
  "energy": 7,
  "notes": "Feeling great today"
}
\`\`\`

**Required Fields:**
- `mood` (integer, 1-10): Mood rating
- `anxiety` (integer, 0-10): Anxiety level
- `energy` (integer, 1-10): Energy level

**Optional Fields:**
- `notes` (string): Additional notes about the mood entry

**Example Response:**

\`\`\`json
{
  "success": true,
  "data": {
    "id": "123",
    "date": "2024-01-15",
    "mood": 8,
    "anxiety": 3,
    "energy": 7,
    "notes": "Feeling great today",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
\`\`\`

### Sleep Tracking

#### Get Sleep Entries

\`\`\`http
GET /api/v1/sleep
\`\`\`

Retrieve sleep tracking entries.

**Query Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| startDate | string | Filter by start date (YYYY-MM-DD) | - |
| endDate | string | Filter by end date (YYYY-MM-DD) | - |
| limit | integer | Number of results to return | 100 |

**Example Response:**

\`\`\`json
{
  "data": [
    {
      "id": "1",
      "date": "2024-01-15",
      "bedtime": "23:00",
      "wakeTime": "07:30",
      "quality": 8,
      "duration": 8.5,
      "notes": "Slept well",
      "timestamp": "2024-01-15T07:30:00Z"
    }
  ],
  "meta": {
    "total": 1,
    "limit": 100
  }
}
\`\`\`

#### Create Sleep Entry

\`\`\`http
POST /api/v1/sleep
\`\`\`

Create a new sleep tracking entry.

**Request Body:**

\`\`\`json
{
  "bedtime": "23:00",
  "wakeTime": "07:30",
  "quality": 8,
  "notes": "Slept well"
}
\`\`\`

**Required Fields:**
- `bedtime` (string, HH:MM): Time went to bed
- `wakeTime` (string, HH:MM): Time woke up
- `quality` (integer, 1-10): Sleep quality rating

**Optional Fields:**
- `notes` (string): Additional notes about sleep

### Medications

#### Get Medications

\`\`\`http
GET /api/v1/medications
\`\`\`

Retrieve all medications and their adherence data.

**Example Response:**

\`\`\`json
{
  "data": [
    {
      "id": "1",
      "name": "Medication A",
      "dosage": "10mg",
      "frequency": "Daily",
      "time": "09:00",
      "active": true,
      "adherence": 95,
      "lastTaken": "2024-01-15T09:00:00Z"
    }
  ],
  "meta": {
    "total": 1
  }
}
\`\`\`

#### Create Medication

\`\`\`http
POST /api/v1/medications
\`\`\`

Add a new medication to track.

**Request Body:**

\`\`\`json
{
  "name": "Medication A",
  "dosage": "10mg",
  "frequency": "Daily",
  "time": "09:00"
}
\`\`\`

**Required Fields:**
- `name` (string): Medication name
- `dosage` (string): Dosage amount
- `frequency` (string): How often to take (Daily, Weekly, Monthly, As Needed)

**Optional Fields:**
- `time` (string, HH:MM): Reminder time (default: 09:00)

### Analytics

#### Get Analytics

\`\`\`http
GET /api/v1/analytics
\`\`\`

Retrieve analytics and insights about your wellness data.

**Query Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| period | string | Time period (week, month, year) | week |

**Example Response:**

\`\`\`json
{
  "period": "week",
  "summary": {
    "averageMood": 7.5,
    "averageAnxiety": 3.2,
    "averageEnergy": 7.8,
    "averageSleep": 7.5,
    "totalEntries": 45,
    "medicationAdherence": 92
  },
  "trends": {
    "mood": [7, 8, 7, 9, 8, 7, 8],
    "anxiety": [3, 2, 4, 2, 3, 3, 2],
    "energy": [8, 7, 8, 9, 7, 8, 8],
    "sleep": [7, 8, 7, 8, 7, 7, 8]
  },
  "correlations": {
    "sleepMood": 0.75,
    "anxietyMood": -0.65,
    "energyMood": 0.82
  },
  "insights": [
    "Your mood improves significantly with better sleep quality",
    "Anxiety levels are lowest on days with regular exercise",
    "Energy levels correlate strongly with mood"
  ]
}
\`\`\`

## Webhooks

Configure webhooks to receive real-time updates when your data changes.

### Webhook Events

- `mood.created`: Triggered when a new mood entry is created
- `sleep.created`: Triggered when a new sleep entry is created
- `medication.taken`: Triggered when a medication is marked as taken
- `medication.missed`: Triggered when a medication reminder is missed

### Webhook Payload

\`\`\`json
{
  "event": "mood.created",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "id": "123",
    "mood": 8,
    "anxiety": 3,
    "energy": 7
  }
}
\`\`\`

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

**Error Response Format:**

\`\`\`json
{
  "error": "Invalid request",
  "message": "Missing required field: mood",
  "code": "VALIDATION_ERROR"
}
\`\`\`

## Best Practices

1. **Secure Your API Keys**: Never expose API keys in client-side code or public repositories
2. **Handle Rate Limits**: Implement exponential backoff when rate limits are reached
3. **Cache Responses**: Cache analytics data to reduce API calls
4. **Use Webhooks**: For real-time updates, use webhooks instead of polling
5. **Validate Data**: Always validate data before sending to the API

## Support

For API support or questions:
- Email: support@mindtrack.app
- Documentation: https://docs.mindtrack.app
- Status Page: https://status.mindtrack.app

## Changelog

### v1.0.0 (2024-01-15)
- Initial API release
- Mood, sleep, medication, and analytics endpoints
- Webhook support
- API key authentication
