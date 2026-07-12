import { sqliteTable, integer, text, real } from 'drizzle-orm/sqlite-core';

// Health tracking tables
export const healthLogs = sqliteTable('health_logs', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  userId: text('user_id').notNull(),
  date: text('date').notNull(),
  sleepStartTime: text('sleep_start_time'),
  sleepEndTime: text('sleep_end_time'),
  sleepDurationHours: real('sleep_duration_hours'),
  sleepQualityScore: integer('sleep_quality_score'),
  moodScore: integer('mood_score'),
  stressScore: integer('stress_score'),
  journalEntry: text('journal_entry'),
  voiceNotePath: text('voice_note_path'),
  giFlare: real('gi_flare'),
  skinFlare: real('skin_flare'),
  migraine: real('migraine'),
  createdAt: text('created_at').notNull(),
});

export const userProfiles = sqliteTable('user_profiles', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  userId: text('user_id').notNull().unique(),
  age: integer('age'),
  gender: text('gender'),
  heightCm: real('height_cm'),
  weightKg: real('weight_kg'),
  bmi: real('bmi'),
  knownConditions: text('known_conditions', { mode: 'json' }),
  medications: text('medications', { mode: 'json' }),
  allergies: text('allergies', { mode: 'json' }),
  activityLevel: text('activity_level'),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
});

export const userBadges = sqliteTable('user_badges', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  userId: text('user_id').notNull(),
  badgeId: text('badge_id').notNull(),
  badgeName: text('badge_name').notNull(),
  badgeDescription: text('badge_description'),
  unlockedAt: text('unlocked_at').notNull(),
});

export const userGoals = sqliteTable('user_goals', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  userId: text('user_id').notNull(),
  goalType: text('goal_type').notNull(),
  goalName: text('goal_name').notNull(),
  targetValue: real('target_value'),
  currentValue: real('current_value'),
  progressPercentage: real('progress_percentage'),
  status: text('status').notNull().default('active'),
  createdAt: text('created_at').notNull(),
});