CREATE TABLE `health_logs` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`user_id` text NOT NULL,
	`date` text NOT NULL,
	`sleep_start_time` text,
	`sleep_end_time` text,
	`sleep_duration_hours` real,
	`sleep_quality_score` integer,
	`mood_score` integer,
	`stress_score` integer,
	`journal_entry` text,
	`voice_note_path` text,
	`gi_flare` real,
	`skin_flare` real,
	`migraine` real,
	`created_at` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `user_badges` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`user_id` text NOT NULL,
	`badge_id` text NOT NULL,
	`badge_name` text NOT NULL,
	`badge_description` text,
	`unlocked_at` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `user_goals` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`user_id` text NOT NULL,
	`goal_type` text NOT NULL,
	`goal_name` text NOT NULL,
	`target_value` real,
	`current_value` real,
	`progress_percentage` real,
	`status` text DEFAULT 'active' NOT NULL,
	`created_at` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `user_profiles` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`user_id` text NOT NULL,
	`age` integer,
	`gender` text,
	`height_cm` real,
	`weight_kg` real,
	`bmi` real,
	`known_conditions` text,
	`medications` text,
	`allergies` text,
	`activity_level` text,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `user_profiles_user_id_unique` ON `user_profiles` (`user_id`);