import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/db';
import { userBadges } from '@/db/schema';
import { eq, like, and, or, desc, asc, SQL } from 'drizzle-orm';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');
    const userId = searchParams.get('userId');
    const badgeId = searchParams.get('badgeId');
    const limit = Math.min(parseInt(searchParams.get('limit') || '10'), 100);
    const offset = parseInt(searchParams.get('offset') || '0');
    const search = searchParams.get('search');
    const sort = searchParams.get('sort') || 'unlockedAt';
    const order = searchParams.get('order') || 'desc';

    // Single record fetch by ID
    if (id) {
      if (!id || isNaN(parseInt(id))) {
        return NextResponse.json({ 
          error: "Valid ID is required",
          code: "INVALID_ID" 
        }, { status: 400 });
      }

      const badge = await db.select()
        .from(userBadges)
        .where(eq(userBadges.id, parseInt(id)))
        .limit(1);

      if (badge.length === 0) {
        return NextResponse.json({ error: 'User badge not found' }, { status: 404 });
      }

      return NextResponse.json(badge[0]);
    }

    // List with filtering
    const conditions: SQL[] = [];

    // Filter by userId
    if (userId) {
      conditions.push(eq(userBadges.userId, userId));
    }

    // Filter by badgeId
    if (badgeId) {
      conditions.push(eq(userBadges.badgeId, badgeId));
    }

    // Search across badge name and description
    if (search) {
      const searchCondition = or(
        like(userBadges.badgeName, `%${search}%`),
        like(userBadges.badgeDescription, `%${search}%`)
      );
      if (searchCondition) {
        conditions.push(searchCondition);
      }
    }

    // Build query conditionally - avoid reassignment to prevent type issues
    const sortField = sort === 'badgeName' ? userBadges.badgeName :
                     sort === 'badgeId' ? userBadges.badgeId :
                     sort === 'userId' ? userBadges.userId :
                     userBadges.unlockedAt;

    const orderDirection = order === 'asc' ? asc(sortField) : desc(sortField);
    const baseQuery = db.select().from(userBadges);

    if (conditions.length > 0) {
      const whereCondition = conditions.length === 1 ? conditions[0] : and(...conditions);
      const queryWithWhere = baseQuery.where(whereCondition);
      const queryWithOrder = queryWithWhere.orderBy(orderDirection);
      const badges = await queryWithOrder.limit(limit).offset(offset);
      return NextResponse.json(badges);
    } else {
      const queryWithOrder = baseQuery.orderBy(orderDirection);
      const badges = await queryWithOrder.limit(limit).offset(offset);
      return NextResponse.json(badges);
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
    const { userId, badgeId, badgeName, badgeDescription } = requestBody;

    // Validate required fields
    if (!userId) {
      return NextResponse.json({ 
        error: "userId is required",
        code: "MISSING_USER_ID" 
      }, { status: 400 });
    }

    if (!badgeId) {
      return NextResponse.json({ 
        error: "badgeId is required",
        code: "MISSING_BADGE_ID" 
      }, { status: 400 });
    }

    if (!badgeName) {
      return NextResponse.json({ 
        error: "badgeName is required",
        code: "MISSING_BADGE_NAME" 
      }, { status: 400 });
    }

    // Validate badgeId format (alphanumeric with underscores)
    const badgeIdRegex = /^[a-zA-Z0-9_]+$/;
    if (!badgeIdRegex.test(badgeId)) {
      return NextResponse.json({ 
        error: "badgeId must contain only alphanumeric characters and underscores",
        code: "INVALID_BADGE_ID_FORMAT" 
      }, { status: 400 });
    }

    // Check for duplicate badge award to same user
    const existingBadge = await db.select()
      .from(userBadges)
      .where(and(
        eq(userBadges.userId, userId),
        eq(userBadges.badgeId, badgeId)
      ))
      .limit(1);

    if (existingBadge.length > 0) {
      return NextResponse.json({ 
        error: "Badge already awarded to this user",
        code: "DUPLICATE_BADGE_AWARD" 
      }, { status: 400 });
    }

    // Sanitize inputs
    const sanitizedData = {
      userId: userId.trim(),
      badgeId: badgeId.toLowerCase().trim(),
      badgeName: badgeName.trim(),
      badgeDescription: badgeDescription ? badgeDescription.trim() : null,
      unlockedAt: new Date().toISOString()
    };

    const newBadge = await db.insert(userBadges)
      .values(sanitizedData)
      .returning();

    return NextResponse.json(newBadge[0], { status: 201 });

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
    const { badgeId, badgeName, badgeDescription } = requestBody;

    // Check if record exists
    const existingBadge = await db.select()
      .from(userBadges)
      .where(eq(userBadges.id, parseInt(id)))
      .limit(1);

    if (existingBadge.length === 0) {
      return NextResponse.json({ error: 'User badge not found' }, { status: 404 });
    }

    // Validate badgeId format if provided
    if (badgeId && !/^[a-zA-Z0-9_]+$/.test(badgeId)) {
      return NextResponse.json({ 
        error: "badgeId must contain only alphanumeric characters and underscores",
        code: "INVALID_BADGE_ID_FORMAT" 
      }, { status: 400 });
    }

    // Check for duplicate badge award if badgeId is being changed
    if (badgeId && badgeId !== existingBadge[0].badgeId) {
      const duplicateBadge = await db.select()
        .from(userBadges)
        .where(and(
          eq(userBadges.userId, existingBadge[0].userId),
          eq(userBadges.badgeId, badgeId)
        ))
        .limit(1);

      if (duplicateBadge.length > 0) {
        return NextResponse.json({ 
          error: "Badge already awarded to this user",
          code: "DUPLICATE_BADGE_AWARD" 
        }, { status: 400 });
      }
    }

    // Prepare updates
    const updates: any = {};
    
    if (badgeId !== undefined) {
      updates.badgeId = badgeId.toLowerCase().trim();
    }
    if (badgeName !== undefined) {
      updates.badgeName = badgeName.trim();
    }
    if (badgeDescription !== undefined) {
      updates.badgeDescription = badgeDescription ? badgeDescription.trim() : null;
    }

    const updatedBadge = await db.update(userBadges)
      .set(updates)
      .where(eq(userBadges.id, parseInt(id)))
      .returning();

    return NextResponse.json(updatedBadge[0]);

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
    const existingBadge = await db.select()
      .from(userBadges)
      .where(eq(userBadges.id, parseInt(id)))
      .limit(1);

    if (existingBadge.length === 0) {
      return NextResponse.json({ error: 'User badge not found' }, { status: 404 });
    }

    const deletedBadge = await db.delete(userBadges)
      .where(eq(userBadges.id, parseInt(id)))
      .returning();

    return NextResponse.json({
      message: 'User badge deleted successfully',
      deletedBadge: deletedBadge[0]
    });

  } catch (error) {
    console.error('DELETE error:', error);
    return NextResponse.json({ 
      error: 'Internal server error: ' + error 
    }, { status: 500 });
  }
}