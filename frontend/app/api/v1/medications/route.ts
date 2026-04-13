import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  const apiKey = request.headers.get("x-api-key")

  if (!apiKey) {
    return NextResponse.json({ error: "API key required" }, { status: 401 })
  }

  const medicationsData = {
    data: [
      {
        id: "1",
        name: "Medication A",
        dosage: "10mg",
        frequency: "Daily",
        time: "09:00",
        active: true,
        adherence: 95,
        lastTaken: new Date().toISOString(),
      },
    ],
    meta: {
      total: 1,
    },
  }

  return NextResponse.json(medicationsData)
}

export async function POST(request: NextRequest) {
  const apiKey = request.headers.get("x-api-key")

  if (!apiKey) {
    return NextResponse.json({ error: "API key required" }, { status: 401 })
  }

  try {
    const body = await request.json()
    const { name, dosage, frequency, time } = body

    if (!name || !dosage || !frequency) {
      return NextResponse.json({ error: "Missing required fields: name, dosage, frequency" }, { status: 400 })
    }

    const newMedication = {
      id: Date.now().toString(),
      name,
      dosage,
      frequency,
      time: time || "09:00",
      active: true,
      adherence: 100,
      createdAt: new Date().toISOString(),
    }

    return NextResponse.json({ success: true, data: newMedication }, { status: 201 })
  } catch (error) {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 })
  }
}
