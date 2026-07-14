import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/db';
import { userProfiles } from '@/db/schema';
import { eq, like, and, or, desc, asc, SQL } from 'drizzle-orm';

function calculateBMI(heightCm: number, weightKg: number): number {
  if (!heightCm || !weightKg || heightCm <= 0 || weightKg <= 0) {
    return 0;
  }
  const heightM = heightCm / 100;
  return parseFloat((weightKg / (heightM * heightM)).toFixed(2));
}

function validateJsonArray(value: any): boolean {
  if (!value) return true;
  try {
    const parsed = typeof value === 'string' ? JSON.parse(value) : value;
    return Array.isArray(parsed);
  } catch {
    return false;
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');
    const userId = searchParams.get('userId');
    const limit = Math.min(parseInt(searchParams.get('limit') || '10'), 100);
    const offset = parseInt(searchParams.get('offset') || '0');
    const search = searchParams.get('search');
    const sort = searchParams.get('sort') || 'createdAt';
    const order = searchParams.get('order') || 'desc';

    // Get single record by ID
    if (id) {
      if (!id || isNaN(parseInt(id))) {
        return NextResponse.json({ 
          error: "Valid ID is required",
          code: "INVALID_ID" 
        }, { status: 400 });
      }

      const record = await db.select()
        .from(userProfiles)
        .where(eq(userProfiles.id, parseInt(id)))
        .limit(1);

      if (record.length === 0) {
        return NextResponse.json({ error: 'User profile not found' }, { status: 404 });
      }

      return NextResponse.json(record[0]);
    }

    // List records with filtering and pagination
    const conditions: SQL[] = [];
    
    if (userId) {
      conditions.push(eq(userProfiles.userId, userId));
    }

    if (search) {
      const searchCondition = or(
        like(userProfiles.userId, `%${search}%`),
        like(userProfiles.gender, `%${search}%`),
        like(userProfiles.activityLevel, `%${search}%`)
      );
      if (searchCondition) {
        conditions.push(searchCondition);
      }
    }

    // Build query conditionally - avoid reassignment to prevent type issues
    const orderBy = order === 'asc' ? asc : desc;
    const sortColumn = sort === 'userId' ? userProfiles.userId :
                      sort === 'age' ? userProfiles.age :
                      sort === 'bmi' ? userProfiles.bmi :
                      sort === 'updatedAt' ? userProfiles.updatedAt :
                      userProfiles.createdAt;

    const baseQuery = db.select().from(userProfiles);

    if (conditions.length > 0) {
      const whereCondition = conditions.length === 1 ? conditions[0] : and(...conditions);
      const queryWithWhere = baseQuery.where(whereCondition);
      const queryWithOrder = queryWithWhere.orderBy(orderBy(sortColumn));
      const results = await queryWithOrder.limit(limit).offset(offset);
      return NextResponse.json(results);
    } else {
      const queryWithOrder = baseQuery.orderBy(orderBy(sortColumn));
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
    const body = await request.json();
    const { 
      userId, 
      age, 
      gender, 
      heightCm, 
      weightKg, 
      knownConditions, 
      medications, 
      allergies, 
      activityLevel 
    } = body;

    // Validate required fields
    if (!userId) {
      return NextResponse.json({ 
        error: "userId is required",
        code: "MISSING_REQUIRED_FIELD" 
      }, { status: 400 });
    }

    // Validate age
    if (age !== undefined && age !== null && (typeof age !== 'number' || age <= 0 || !Number.isInteger(age))) {
      return NextResponse.json({ 
        error: "Age must be a positive integer",
        code: "INVALID_AGE" 
      }, { status: 400 });
    }

    // Validate height and weight
    if (heightCm !== undefined && heightCm !== null && (typeof heightCm !== 'number' || heightCm <= 0)) {
      return NextResponse.json({ 
        error: "Height must be a positive number",
        code: "INVALID_HEIGHT" 
      }, { status: 400 });
    }

    if (weightKg !== undefined && weightKg !== null && (typeof weightKg !== 'number' || weightKg <= 0)) {
      return NextResponse.json({ 
        error: "Weight must be a positive number",
        code: "INVALID_WEIGHT" 
      }, { status: 400 });
    }

    // Validate JSON arrays
    if (knownConditions && !validateJsonArray(knownConditions)) {
      return NextResponse.json({ 
        error: "knownConditions must be a valid JSON array",
        code: "INVALID_JSON_ARRAY" 
      }, { status: 400 });
    }

    if (medications && !validateJsonArray(medications)) {
      return NextResponse.json({ 
        error: "medications must be a valid JSON array",
        code: "INVALID_JSON_ARRAY" 
      }, { status: 400 });
    }

    if (allergies && !validateJsonArray(allergies)) {
      return NextResponse.json({ 
        error: "allergies must be a valid JSON array",
        code: "INVALID_JSON_ARRAY" 
      }, { status: 400 });
    }

    // Check if userId already exists
    const existingProfile = await db.select()
      .from(userProfiles)
      .where(eq(userProfiles.userId, userId))
      .limit(1);

    if (existingProfile.length > 0) {
      return NextResponse.json({ 
        error: "User profile with this userId already exists",
        code: "DUPLICATE_USER_ID" 
      }, { status: 400 });
    }

    // Calculate BMI
    const bmi = calculateBMI(heightCm, weightKg);

    // Prepare insert data
    const now = new Date().toISOString();
    const insertData = {
      userId: userId.trim(),
      age: age || null,
      gender: gender?.trim() || null,
      heightCm: heightCm || null,
      weightKg: weightKg || null,
      bmi: bmi || null,
      knownConditions: knownConditions ? JSON.stringify(knownConditions) : null,
      medications: medications ? JSON.stringify(medications) : null,
      allergies: allergies ? JSON.stringify(allergies) : null,
      activityLevel: activityLevel?.trim() || null,
      createdAt: now,
      updatedAt: now
    };

    const newRecord = await db.insert(userProfiles)
      .values(insertData)
      .returning();

    return NextResponse.json(newRecord[0], { status: 201 });

  } catch (error) {
    console.error('POST error:', error);
    if (error instanceof Error && error.message.includes('UNIQUE constraint')) {
      return NextResponse.json({ 
        error: "User profile with this userId already exists",
        code: "DUPLICATE_USER_ID" 
      }, { status: 400 });
    }
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

    const body = await request.json();
    const { 
      userId, 
      age, 
      gender, 
      heightCm, 
      weightKg, 
      knownConditions, 
      medications, 
      allergies, 
      activityLevel 
    } = body;

    // Check if record exists
    const existingRecord = await db.select()
      .from(userProfiles)
      .where(eq(userProfiles.id, parseInt(id)))
      .limit(1);

    if (existingRecord.length === 0) {
      return NextResponse.json({ error: 'User profile not found' }, { status: 404 });
    }

    // If updating userId, check for uniqueness
    if (userId && userId !== existingRecord[0].userId) {
      const duplicateCheck = await db.select()
        .from(userProfiles)
        .where(eq(userProfiles.userId, userId))
        .limit(1);

      if (duplicateCheck.length > 0) {
        return NextResponse.json({ 
          error: "User profile with this userId already exists",
          code: "DUPLICATE_USER_ID" 
        }, { status: 400 });
      }
    }

    // Validate age
    if (age !== undefined && age !== null && (typeof age !== 'number' || age <= 0 || !Number.isInteger(age))) {
      return NextResponse.json({ 
        error: "Age must be a positive integer",
        code: "INVALID_AGE" 
      }, { status: 400 });
    }

    // Validate height and weight
    if (heightCm !== undefined && heightCm !== null && (typeof heightCm !== 'number' || heightCm <= 0)) {
      return NextResponse.json({ 
        error: "Height must be a positive number",
        code: "INVALID_HEIGHT" 
      }, { status: 400 });
    }

    if (weightKg !== undefined && weightKg !== null && (typeof weightKg !== 'number' || weightKg <= 0)) {
      return NextResponse.json({ 
        error: "Weight must be a positive number",
        code: "INVALID_WEIGHT" 
      }, { status: 400 });
    }

    // Validate JSON arrays
    if (knownConditions !== undefined && !validateJsonArray(knownConditions)) {
      return NextResponse.json({ 
        error: "knownConditions must be a valid JSON array",
        code: "INVALID_JSON_ARRAY" 
      }, { status: 400 });
    }

    if (medications !== undefined && !validateJsonArray(medications)) {
      return NextResponse.json({ 
        error: "medications must be a valid JSON array",
        code: "INVALID_JSON_ARRAY" 
      }, { status: 400 });
    }

    if (allergies !== undefined && !validateJsonArray(allergies)) {
      return NextResponse.json({ 
        error: "allergies must be a valid JSON array",
        code: "INVALID_JSON_ARRAY" 
      }, { status: 400 });
    }

    // Prepare update data
    const updates: any = {
      updatedAt: new Date().toISOString()
    };

    if (userId !== undefined) updates.userId = userId.trim();
    if (age !== undefined) updates.age = age;
    if (gender !== undefined) updates.gender = gender?.trim() || null;
    if (heightCm !== undefined) updates.heightCm = heightCm;
    if (weightKg !== undefined) updates.weightKg = weightKg;
    if (knownConditions !== undefined) updates.knownConditions = knownConditions ? JSON.stringify(knownConditions) : null;
    if (medications !== undefined) updates.medications = medications ? JSON.stringify(medications) : null;
    if (allergies !== undefined) updates.allergies = allergies ? JSON.stringify(allergies) : null;
    if (activityLevel !== undefined) updates.activityLevel = activityLevel?.trim() || null;

    // Recalculate BMI if height or weight changed
    const finalHeightCm = heightCm !== undefined ? heightCm : existingRecord[0].heightCm;
    const finalWeightKg = weightKg !== undefined ? weightKg : existingRecord[0].weightKg;
    
    if (heightCm !== undefined || weightKg !== undefined) {
      updates.bmi = calculateBMI(finalHeightCm, finalWeightKg);
    }

    const updated = await db.update(userProfiles)
      .set(updates)
      .where(eq(userProfiles.id, parseInt(id)))
      .returning();

    if (updated.length === 0) {
      return NextResponse.json({ error: 'User profile not found' }, { status: 404 });
    }

    return NextResponse.json(updated[0]);

  } catch (error) {
    console.error('PUT error:', error);
    if (error instanceof Error && error.message.includes('UNIQUE constraint')) {
      return NextResponse.json({ 
        error: "User profile with this userId already exists",
        code: "DUPLICATE_USER_ID" 
      }, { status: 400 });
    }
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
      .from(userProfiles)
      .where(eq(userProfiles.id, parseInt(id)))
      .limit(1);

    if (existingRecord.length === 0) {
      return NextResponse.json({ error: 'User profile not found' }, { status: 404 });
    }

    const deleted = await db.delete(userProfiles)
      .where(eq(userProfiles.id, parseInt(id)))
      .returning();

    if (deleted.length === 0) {
      return NextResponse.json({ error: 'User profile not found' }, { status: 404 });
    }

    return NextResponse.json({
      message: 'User profile deleted successfully',
      deletedRecord: deleted[0]
    });

  } catch (error) {
    console.error('DELETE error:', error);
    return NextResponse.json({ 
      error: 'Internal server error: ' + error 
    }, { status: 500 });
  }
}