import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  const apiKey = request.headers.get("x-api-key")

  if (!apiKey) {
    return NextResponse.json({ error: "API key required" }, { status: 401 })
  }

  const searchParams = request.nextUrl.searchParams
  const startDate = searchParams.get("startDate")
  const endDate = searchParams.get("endDate")
  const limit = searchParams.get("limit") || "100"

  const sleepData = {
    data: [
      {
        id: "1",
        date: "2024-01-15",
        bedtime: "23:00",
        wakeTime: "07:30",
        quality: 8,
        duration: 8.5,
        notes: "Slept well",
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

  return NextResponse.json(sleepData)
}

export async function POST(request: NextRequest) {
  const apiKey = request.headers.get("x-api-key")

  if (!apiKey) {
    return NextResponse.json({ error: "API key required" }, { status: 401 })
  }

  try {
    const body = await request.json()
    const { bedtime, wakeTime, quality, notes } = body

    if (!bedtime || !wakeTime || quality === undefined) {
      return NextResponse.json({ error: "Missing required fields: bedtime, wakeTime, quality" }, { status: 400 })
    }

    const newEntry = {
      id: Date.now().toString(),
      date: new Date().toISOString().split("T")[0],
      bedtime,
      wakeTime,
      quality,
      notes: notes || "",
      timestamp: new Date().toISOString(),
    }

    return NextResponse.json({ success: true, data: newEntry }, { status: 201 })
  } catch (error) {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 })
  }
}
