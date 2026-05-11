import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  const apiKey = request.headers.get("x-api-key")

  if (!apiKey) {
    return NextResponse.json({ error: "API key required" }, { status: 401 })
  }

  // Get query parameters for filtering
  const searchParams = request.nextUrl.searchParams
  const startDate = searchParams.get("startDate")
  const endDate = searchParams.get("endDate")
  const limit = searchParams.get("limit") || "100"

  // In a real app, fetch from database
  // For now, return sample data structure
  const moodData = {
    data: [
      {
        id: "1",
        date: "2024-01-15",
        mood: 8,
        anxiety: 3,
        energy: 7,
        notes: "Feeling great today",
        timestamp: new Date().toISOString(),
      },
    ],
    meta: {
      total: 1,
      limit: Number.parseInt(limit),
      startDate,
      endDate,
    },
  }

  return NextResponse.json(moodData)
}

export async function POST(request: NextRequest) {
  const apiKey = request.headers.get("x-api-key")

  if (!apiKey) {
    return NextResponse.json({ error: "API key required" }, { status: 401 })
  }

  try {
    const body = await request.json()
    const { mood, anxiety, energy, notes } = body

    // Validate required fields
    if (mood === undefined || anxiety === undefined || energy === undefined) {
      return NextResponse.json({ error: "Missing required fields: mood, anxiety, energy" }, { status: 400 })
    }

    // In a real app, save to database
    const newEntry = {
      id: Date.now().toString(),
      date: new Date().toISOString().split("T")[0],
      mood,
      anxiety,
      energy,
      notes: notes || "",
      timestamp: new Date().toISOString(),
    }

    return NextResponse.json({ success: true, data: newEntry }, { status: 201 })
  } catch (error) {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 })
  }
}
