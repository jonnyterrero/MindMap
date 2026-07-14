import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/db';
import { userGoals } from '@/db/schema';
import { eq, like, and, or, desc, asc, SQL } from 'drizzle-orm';

const VALID_STATUSES = ['active', 'achieved', 'at_risk'];

function calculateProgressPercentage(currentValue: number | null, targetValue: number | null): number {
  if (!currentValue || !targetValue || targetValue <= 0) return 0;
  const percentage = (currentValue / targetValue) * 100;
  return Math.min(percentage, 100);
}

function determineStatus(progressPercentage: number, currentStatus?: string): string {
  if (progressPercentage >= 100) return 'achieved';
  return currentStatus || 'active';
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');
    const userId = searchParams.get('userId');
    const status = searchParams.get('status');
    const goalType = searchParams.get('goalType');
    const limit = Math.min(parseInt(searchParams.get('limit') || '10'), 100);
    const offset = parseInt(searchParams.get('offset') || '0');
    const search = searchParams.get('search');
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
        .from(userGoals)
        .where(eq(userGoals.id, parseInt(id)))
        .limit(1);

      if (record.length === 0) {
        return NextResponse.json({ error: 'User goal not found' }, { status: 404 });
      }

      return NextResponse.json(record[0]);
    }

    // List with filters
    const conditions: SQL[] = [];

    if (userId) {
      conditions.push(eq(userGoals.userId, userId));
    }

    if (status) {
      if (!VALID_STATUSES.includes(status)) {
        return NextResponse.json({ 
          error: `Invalid status. Must be one of: ${VALID_STATUSES.join(', ')}`,
          code: "INVALID_STATUS" 
        }, { status: 400 });
      }
      conditions.push(eq(userGoals.status, status));
    }

    if (goalType) {
      conditions.push(eq(userGoals.goalType, goalType));
    }

    if (search) {
      const searchCondition = or(
        like(userGoals.goalName, `%${search}%`),
        like(userGoals.goalType, `%${search}%`)
      );
      if (searchCondition) {
        conditions.push(searchCondition);
      }
    }

    // Build query conditionally - avoid reassignment to prevent type issues
    const orderDirection = order === 'asc' ? asc : desc;
    const sortColumn = sort === 'createdAt' ? userGoals.createdAt :
                      sort === 'goalName' ? userGoals.goalName :
                      sort === 'status' ? userGoals.status :
                      sort === 'progressPercentage' ? userGoals.progressPercentage :
                      userGoals.createdAt;

    const baseQuery = db.select().from(userGoals);

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
    const { userId, goalType, goalName, targetValue, currentValue, status } = requestBody;

    // Validate required fields
    if (!userId) {
      return NextResponse.json({ 
        error: "User ID is required",
        code: "MISSING_USER_ID" 
      }, { status: 400 });
    }

    if (!goalType) {
      return NextResponse.json({ 
        error: "Goal type is required",
        code: "MISSING_GOAL_TYPE" 
      }, { status: 400 });
    }

    if (!goalName) {
      return NextResponse.json({ 
        error: "Goal name is required",
        code: "MISSING_GOAL_NAME" 
      }, { status: 400 });
    }

    // Validate status if provided
    if (status && !VALID_STATUSES.includes(status)) {
      return NextResponse.json({ 
        error: `Invalid status. Must be one of: ${VALID_STATUSES.join(', ')}`,
        code: "INVALID_STATUS" 
      }, { status: 400 });
    }

    // Validate values are positive if provided
    if (targetValue !== undefined && targetValue !== null && targetValue < 0) {
      return NextResponse.json({ 
        error: "Target value must be positive",
        code: "INVALID_TARGET_VALUE" 
      }, { status: 400 });
    }

    if (currentValue !== undefined && currentValue !== null && currentValue < 0) {
      return NextResponse.json({ 
        error: "Current value must be positive",
        code: "INVALID_CURRENT_VALUE" 
      }, { status: 400 });
    }

    // Calculate progress percentage
    const progressPercentage = calculateProgressPercentage(currentValue || 0, targetValue || 1);
    
    // Determine status based on progress
    const finalStatus = determineStatus(progressPercentage, status);

    const insertData = {
      userId: userId.trim(),
      goalType: goalType.trim(),
      goalName: goalName.trim(),
      targetValue: targetValue || null,
      currentValue: currentValue || 0,
      progressPercentage,
      status: finalStatus,
      createdAt: new Date().toISOString()
    };

    const newRecord = await db.insert(userGoals)
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

    const updates = await request.json();

    // Check if record exists
    const existing = await db.select()
      .from(userGoals)
      .where(eq(userGoals.id, parseInt(id)))
      .limit(1);

    if (existing.length === 0) {
      return NextResponse.json({ error: 'User goal not found' }, { status: 404 });
    }

    const currentRecord = existing[0];

    // Validate status if provided
    if (updates.status && !VALID_STATUSES.includes(updates.status)) {
      return NextResponse.json({ 
        error: `Invalid status. Must be one of: ${VALID_STATUSES.join(', ')}`,
        code: "INVALID_STATUS" 
      }, { status: 400 });
    }

    // Validate values are positive if provided
    if (updates.targetValue !== undefined && updates.targetValue !== null && updates.targetValue < 0) {
      return NextResponse.json({ 
        error: "Target value must be positive",
        code: "INVALID_TARGET_VALUE" 
      }, { status: 400 });
    }

    if (updates.currentValue !== undefined && updates.currentValue !== null && updates.currentValue < 0) {
      return NextResponse.json({ 
        error: "Current value must be positive",
        code: "INVALID_CURRENT_VALUE" 
      }, { status: 400 });
    }

    // Calculate new progress percentage if values changed
    const newTargetValue = updates.targetValue !== undefined ? updates.targetValue : currentRecord.targetValue;
    const newCurrentValue = updates.currentValue !== undefined ? updates.currentValue : currentRecord.currentValue;
    
    let progressPercentage: number = currentRecord.progressPercentage ?? 0;
    if (updates.targetValue !== undefined || updates.currentValue !== undefined) {
      progressPercentage = calculateProgressPercentage(newCurrentValue, newTargetValue);
    }

    // Determine status based on progress (unless explicitly overridden)
    let finalStatus = updates.status || currentRecord.status;
    if (updates.targetValue !== undefined || updates.currentValue !== undefined) {
      finalStatus = determineStatus(progressPercentage, updates.status || currentRecord.status);
    }

    const updateData = {
      ...updates,
      progressPercentage,
      status: finalStatus
    };

    // Remove undefined values
    Object.keys(updateData).forEach(key => {
      if (updateData[key] === undefined) {
        delete updateData[key];
      }
    });

    // Trim string fields
    if (updateData.goalType) updateData.goalType = updateData.goalType.trim();
    if (updateData.goalName) updateData.goalName = updateData.goalName.trim();
    if (updateData.userId) updateData.userId = updateData.userId.trim();

    const updated = await db.update(userGoals)
      .set(updateData)
      .where(eq(userGoals.id, parseInt(id)))
      .returning();

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
    const existing = await db.select()
      .from(userGoals)
      .where(eq(userGoals.id, parseInt(id)))
      .limit(1);

    if (existing.length === 0) {
      return NextResponse.json({ error: 'User goal not found' }, { status: 404 });
    }

    const deleted = await db.delete(userGoals)
      .where(eq(userGoals.id, parseInt(id)))
      .returning();

    return NextResponse.json({ 
      message: 'User goal deleted successfully',
      deletedRecord: deleted[0]
    });

  } catch (error) {
    console.error('DELETE error:', error);
    return NextResponse.json({ 
      error: 'Internal server error: ' + error 
    }, { status: 500 });
  }
}