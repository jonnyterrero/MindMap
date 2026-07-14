import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/db';
import { healthLogs } from '@/db/schema';
import { eq, like, and, or, desc, asc, gte, lte, SQL } from 'drizzle-orm';

// Helper function to validate date format (YYYY-MM-DD)
function isValidDate(dateString: string): boolean {
  const regex = /^\d{4}-\d{2}-\d{2}$/;
  if (!regex.test(dateString)) return false;
  const date = new Date(dateString);
  return date instanceof Date && !isNaN(date.getTime());
}

// Helper function to validate time format (HH:MM)
function isValidTime(timeString: string): boolean {
  const regex = /^([01]?[0-9]|2[0-3]):[0-5][0-9]$/;
  return regex.test(timeString);
}

// Helper function to validate score range
function isValidScore(score: number, min: number, max: number): boolean {
  return Number.isInteger(score) && score >= min && score <= max;
}

// Helper function to validate real score range
function isValidRealScore(score: number, min: number, max: number): boolean {
  return typeof score === 'number' && score >= min && score <= max;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');
    const limit = Math.min(parseInt(searchParams.get('limit') || '10'), 100);
    const offset = parseInt(searchParams.get('offset') || '0');
    const userId = searchParams.get('userId');
    const startDate = searchParams.get('startDate');
    const endDate = searchParams.get('endDate');
    const sort = searchParams.get('sort') || 'createdAt';
    const order = searchParams.get('order') || 'desc';

    // Single record fetch
    if (id) {
      if (!id || isNaN(parseInt(id))) {
        return NextResponse.json({ 
          error: "Valid ID is required",
          code: "INVALID_ID" 
        }, { status: 400 });
      }

      const record = await db.select()
        .from(healthLogs)
        .where(eq(healthLogs.id, parseInt(id)))
        .limit(1);

      if (record.length === 0) {
        return NextResponse.json({ error: 'Health log not found' }, { status: 404 });
      }

      return NextResponse.json(record[0]);
    }

    // List with filtering
    const conditions: SQL[] = [];

    // Filter by userId
    if (userId) {
      conditions.push(eq(healthLogs.userId, userId));
    }

    // Filter by date range
    if (startDate) {
      if (!isValidDate(startDate)) {
        return NextResponse.json({ 
          error: "Invalid start date format. Use YYYY-MM-DD",
          code: "INVALID_START_DATE" 
        }, { status: 400 });
      }
      conditions.push(gte(healthLogs.date, startDate));
    }

    if (endDate) {
      if (!isValidDate(endDate)) {
        return NextResponse.json({ 
          error: "Invalid end date format. Use YYYY-MM-DD",
          code: "INVALID_END_DATE" 
        }, { status: 400 });
      }
      conditions.push(lte(healthLogs.date, endDate));
    }

    // Build and execute query - avoid reassignment to prevent type issues
    const orderDirection = order === 'asc' ? asc : desc;
    const sortColumn = sort === 'date' ? healthLogs.date : healthLogs.createdAt;
    
    // Build query conditionally based on whether we have conditions
    const baseQuery = db.select().from(healthLogs);
    
    if (conditions.length > 0) {
      const whereCondition = conditions.length === 1 ? conditions[0] : and(...conditions);
      const queryWithWhere = baseQuery.where(whereCondition);
      const queryWithOrder = queryWithWhere.orderBy(orderDirection(sortColumn));
      const results = await queryWithOrder.limit(limit).offset(offset);
      return NextResponse.json(results);
    } else {
      const queryWithOrder = baseQuery.orderBy(orderDirection(sortColumn));
      const results = await queryWithOrder.limit(limit).offset(offset);
      return NextResponse.json(results);
    }
  } catch (error) {
    console.error('GET error:', error);
    return NextResponse.json({ 
      error: 'Internal server error: ' + error 
    }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const requestBody = await request.json();
    const {
      userId,
      date,
      sleepStartTime,
      sleepEndTime,
      sleepDurationHours,
      sleepQualityScore,
      moodScore,
      stressScore,
      journalEntry,
      voiceNotePath,
      giFlare,
      skinFlare,
      migraine
    } = requestBody;

    // Validate required fields
    if (!userId) {
      return NextResponse.json({ 
        error: "User ID is required",
        code: "MISSING_USER_ID" 
      }, { status: 400 });
    }

    if (!date) {
      return NextResponse.json({ 
        error: "Date is required",
        code: "MISSING_DATE" 
      }, { status: 400 });
    }

    // Validate date format
    if (!isValidDate(date)) {
      return NextResponse.json({ 
        error: "Invalid date format. Use YYYY-MM-DD",
        code: "INVALID_DATE_FORMAT" 
      }, { status: 400 });
    }

    // Validate sleep times if provided
    if (sleepStartTime && !isValidTime(sleepStartTime)) {
      return NextResponse.json({ 
        error: "Invalid sleep start time format. Use HH:MM",
        code: "INVALID_SLEEP_START_TIME" 
      }, { status: 400 });
    }

    if (sleepEndTime && !isValidTime(sleepEndTime)) {
      return NextResponse.json({ 
        error: "Invalid sleep end time format. Use HH:MM",
        code: "INVALID_SLEEP_END_TIME" 
      }, { status: 400 });
    }

    // Validate score ranges
    if (sleepQualityScore !== undefined && !isValidScore(sleepQualityScore, 1, 10)) {
      return NextResponse.json({ 
        error: "Sleep quality score must be an integer between 1 and 10",
        code: "INVALID_SLEEP_QUALITY_SCORE" 
      }, { status: 400 });
    }

    if (moodScore !== undefined && !isValidScore(moodScore, 1, 10)) {
      return NextResponse.json({ 
        error: "Mood score must be an integer between 1 and 10",
        code: "INVALID_MOOD_SCORE" 
      }, { status: 400 });
    }

    if (stressScore !== undefined && !isValidScore(stressScore, 1, 10)) {
      return NextResponse.json({ 
        error: "Stress score must be an integer between 1 and 10",
        code: "INVALID_STRESS_SCORE" 
      }, { status: 400 });
    }

    if (giFlare !== undefined && !isValidRealScore(giFlare, 0, 10)) {
      return NextResponse.json({ 
        error: "GI flare score must be a number between 0 and 10",
        code: "INVALID_GI_FLARE_SCORE" 
      }, { status: 400 });
    }

    if (skinFlare !== undefined && !isValidRealScore(skinFlare, 0, 10)) {
      return NextResponse.json({ 
        error: "Skin flare score must be a number between 0 and 10",
        code: "INVALID_SKIN_FLARE_SCORE" 
      }, { status: 400 });
    }

    if (migraine !== undefined && !isValidRealScore(migraine, 0, 10)) {
      return NextResponse.json({ 
        error: "Migraine score must be a number between 0 and 10",
        code: "INVALID_MIGRAINE_SCORE" 
      }, { status: 400 });
    }

    const insertData = {
      userId: userId.trim(),
      date,
      sleepStartTime: sleepStartTime || null,
      sleepEndTime: sleepEndTime || null,
      sleepDurationHours: sleepDurationHours || null,
      sleepQualityScore: sleepQualityScore || null,
      moodScore: moodScore || null,
      stressScore: stressScore || null,
      journalEntry: journalEntry?.trim() || null,
      voiceNotePath: voiceNotePath?.trim() || null,
      giFlare: giFlare || null,
      skinFlare: skinFlare || null,
      migraine: migraine || null,
      createdAt: new Date().toISOString()
    };

    const newRecord = await db.insert(healthLogs)
      .values(insertData)
      .returning();

    return NextResponse.json(newRecord[0], { status: 201 });
  } catch (error) {
    console.error('POST error:', error);
    return NextResponse.json({ 
      error: 'Internal server error: ' + error 
    }, { status: 500 });
  }
}

export async function PUT(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (!id || isNaN(parseInt(id))) {
      return NextResponse.json({ 
        error: "Valid ID is required",
        code: "INVALID_ID" 
      }, { status: 400 });
    }

    const requestBody = await request.json();
    const {
      userId,
      date,
      sleepStartTime,
      sleepEndTime,
      sleepDurationHours,
      sleepQualityScore,
      moodScore,
      stressScore,
      journalEntry,
      voiceNotePath,
      giFlare,
      skinFlare,
      migraine
    } = requestBody;

    // Check if record exists
    const existingRecord = await db.select()
      .from(healthLogs)
      .where(eq(healthLogs.id, parseInt(id)))
      .limit(1);

    if (existingRecord.length === 0) {
      return NextResponse.json({ error: 'Health log not found' }, { status: 404 });
    }

    // Validate fields if provided
    if (date && !isValidDate(date)) {
      return NextResponse.json({ 
        error: "Invalid date format. Use YYYY-MM-DD",
        code: "INVALID_DATE_FORMAT" 
      }, { status: 400 });
    }

    if (sleepStartTime && !isValidTime(sleepStartTime)) {
      return NextResponse.json({ 
        error: "Invalid sleep start time format. Use HH:MM",
        code: "INVALID_SLEEP_START_TIME" 
      }, { status: 400 });
    }

    if (sleepEndTime && !isValidTime(sleepEndTime)) {
      return NextResponse.json({ 
        error: "Invalid sleep end time format. Use HH:MM",
        code: "INVALID_SLEEP_END_TIME" 
      }, { status: 400 });
    }

    if (sleepQualityScore !== undefined && !isValidScore(sleepQualityScore, 1, 10)) {
      return NextResponse.json({ 
        error: "Sleep quality score must be an integer between 1 and 10",
        code: "INVALID_SLEEP_QUALITY_SCORE" 
      }, { status: 400 });
    }

    if (moodScore !== undefined && !isValidScore(moodScore, 1, 10)) {
      return NextResponse.json({ 
        error: "Mood score must be an integer between 1 and 10",
        code: "INVALID_MOOD_SCORE" 
      }, { status: 400 });
    }

    if (stressScore !== undefined && !isValidScore(stressScore, 1, 10)) {
      return NextResponse.json({ 
        error: "Stress score must be an integer between 1 and 10",
        code: "INVALID_STRESS_SCORE" 
      }, { status: 400 });
    }

    if (giFlare !== undefined && !isValidRealScore(giFlare, 0, 10)) {
      return NextResponse.json({ 
        error: "GI flare score must be a number between 0 and 10",
        code: "INVALID_GI_FLARE_SCORE" 
      }, { status: 400 });
    }

    if (skinFlare !== undefined && !isValidRealScore(skinFlare, 0, 10)) {
      return NextResponse.json({ 
        error: "Skin flare score must be a number between 0 and 10",
        code: "INVALID_SKIN_FLARE_SCORE" 
      }, { status: 400 });
    }

    if (migraine !== undefined && !isValidRealScore(migraine, 0, 10)) {
      return NextResponse.json({ 
        error: "Migraine score must be a number between 0 and 10",
        code: "INVALID_MIGRAINE_SCORE" 
      }, { status: 400 });
    }

    // Build update object with only provided fields
    const updates: any = {};

    if (userId !== undefined) updates.userId = userId.trim();
    if (date !== undefined) updates.date = date;
    if (sleepStartTime !== undefined) updates.sleepStartTime = sleepStartTime;
    if (sleepEndTime !== undefined) updates.sleepEndTime = sleepEndTime;
    if (sleepDurationHours !== undefined) updates.sleepDurationHours = sleepDurationHours;
    if (sleepQualityScore !== undefined) updates.sleepQualityScore = sleepQualityScore;
    if (moodScore !== undefined) updates.moodScore = moodScore;
    if (stressScore !== undefined) updates.stressScore = stressScore;
    if (journalEntry !== undefined) updates.journalEntry = journalEntry?.trim() || null;
    if (voiceNotePath !== undefined) updates.voiceNotePath = voiceNotePath?.trim() || null;
    if (giFlare !== undefined) updates.giFlare = giFlare;
    if (skinFlare !== undefined) updates.skinFlare = skinFlare;
    if (migraine !== undefined) updates.migraine = migraine;

    const updated = await db.update(healthLogs)
      .set(updates)
      .where(eq(healthLogs.id, parseInt(id)))
      .returning();

    if (updated.length === 0) {
      return NextResponse.json({ error: 'Health log not found' }, { status: 404 });
    }

    return NextResponse.json(updated[0]);
  } catch (error) {
    console.error('PUT error:', error);
    return NextResponse.json({ 
      error: 'Internal server error: ' + error 
    }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (!id || isNaN(parseInt(id))) {
      return NextResponse.json({ 
        error: "Valid ID is required",
        code: "INVALID_ID" 
      }, { status: 400 });
    }

    // Check if record exists
    const existingRecord = await db.select()
      .from(healthLogs)
      .where(eq(healthLogs.id, parseInt(id)))
      .limit(1);

    if (existingRecord.length === 0) {
      return NextResponse.json({ error: 'Health log not found' }, { status: 404 });
    }

    const deleted = await db.delete(healthLogs)
      .where(eq(healthLogs.id, parseInt(id)))
      .returning();

    if (deleted.length === 0) {
      return NextResponse.json({ error: 'Health log not found' }, { status: 404 });
    }

    return NextResponse.json({
      message: 'Health log deleted successfully',
      deletedRecord: deleted[0]
    });
  } catch (error) {
    console.error('DELETE error:', error);
    return NextResponse.json({ 
      error: 'Internal server error: ' + error 
    }, { status: 500 });
  }
}