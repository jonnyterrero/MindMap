-- ============================================================
-- Migration 013: Weather correlation (optional, opt-in)
-- Adds an opt-in daily weather snapshot per user/day so the
-- correlation engine can relate barometric pressure / humidity /
-- temperature to mood, sleep, and migraine. All additive.
--
-- Weather data is fetched from Open-Meteo (no API key). Disabled
-- by default; users opt in and set a location in Settings.
-- ============================================================

BEGIN;

-- Opt-in flag + stored location on the profile.
ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS weather_enabled boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS weather_lat double precision,
  ADD COLUMN IF NOT EXISTS weather_lon double precision,
  ADD COLUMN IF NOT EXISTS weather_label text;

-- One weather snapshot per user per day.
CREATE TABLE IF NOT EXISTS public.mindmap_weather_daily (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  entry_date date NOT NULL,
  temp_max double precision,
  temp_min double precision,
  humidity double precision,
  pressure double precision,
  pressure_change double precision,
  precipitation double precision,
  weather_code integer,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, entry_date)
);

CREATE INDEX IF NOT EXISTS idx_mindmap_weather_daily_user_date
  ON public.mindmap_weather_daily (user_id, entry_date DESC);

ALTER TABLE public.mindmap_weather_daily ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  CREATE POLICY weather_daily_select_own ON public.mindmap_weather_daily
    FOR SELECT USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE POLICY weather_daily_insert_own ON public.mindmap_weather_daily
    FOR INSERT WITH CHECK (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE POLICY weather_daily_update_own ON public.mindmap_weather_daily
    FOR UPDATE USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE POLICY weather_daily_delete_own ON public.mindmap_weather_daily
    FOR DELETE USING (auth.uid() = user_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMIT;
