import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  const apiKey = request.headers.get("x-api-key")

  if (!apiKey) {
    return NextResponse.json({ error: "API key required" }, { status: 401 })
  }

  const searchParams = request.nextUrl.searchParams
  const period = searchParams.get("period") || "week" // week, month, year

  const analyticsData = {
    period,
    summary: {
      averageMood: 7.5,
      averageAnxiety: 3.2,
      averageEnergy: 7.8,
      averageSleep: 7.5,
      totalEntries: 45,
      medicationAdherence: 92,
    },
    trends: {
      mood: [7, 8, 7, 9, 8, 7, 8],
      anxiety: [3, 2, 4, 2, 3, 3, 2],
      energy: [8, 7, 8, 9, 7, 8, 8],
      sleep: [7, 8, 7, 8, 7, 7, 8],
    },
    correlations: {
      sleepMood: 0.75,
      anxietyMood: -0.65,
      energyMood: 0.82,
    },
    insights: [
      "Your mood improves significantly with better sleep quality",
      "Anxiety levels are lowest on days with regular exercise",
      "Energy levels correlate strongly with mood",
    ],
  }

  return NextResponse.json(analyticsData)
}
