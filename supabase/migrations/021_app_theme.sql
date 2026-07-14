-- ============================================================
-- Migration 021: App appearance theme
-- Adds a user-selectable color theme to profiles so the chosen
-- "color mood" follows the user across devices. Additive + idempotent.
--
-- Product notes:
--  * app_theme controls the liquid-glass accent palette (nav, CTAs,
--    selected states, gradient accents). It is purely cosmetic and
--    never affects health data.
--  * Allowed values mirror lib/themes.ts. Defaults to 'aurora'.
-- ============================================================

BEGIN;

ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS app_theme text NOT NULL DEFAULT 'aurora';

DO $$ BEGIN
  ALTER TABLE public.profiles
    ADD CONSTRAINT chk_profiles_app_theme
    CHECK (
      app_theme IN ('aurora','ocean','lavender','rose','graphite','forest')
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

COMMIT;
