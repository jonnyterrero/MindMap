-- ============================================================
-- Migration 012: Guided UX support
-- Adds onboarding state + check-in configuration to profiles, and a
-- daily MindMap Score to mindmap_entries. All additive and idempotent.
--
-- Product notes:
--  * selected_focus is the user's primary tracking goal (onboarding).
--  * selected_checkin_cards controls which sections appear in the
--    guided daily check-in. Defaults to all cards (user can deselect).
--  * mindmap_score (0..100) rewards tracking consistency only. It is
--    never a health score and is never reduced for "bad" health days.
-- ============================================================

BEGIN;

-- ------------------------------------------------------------
-- 1) profiles: onboarding + check-in configuration
-- ------------------------------------------------------------
ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS onboarding_complete boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS onboarding_completed_at timestamptz,
  ADD COLUMN IF NOT EXISTS selected_focus text,
  ADD COLUMN IF NOT EXISTS selected_checkin_cards text[] NOT NULL
    DEFAULT ARRAY['sleep','mood','focus','migraine','medication','routines','journal']::text[];

-- Constrain the focus to the onboarding options (nullable until chosen).
DO $$ BEGIN
  ALTER TABLE public.profiles
    ADD CONSTRAINT chk_profiles_selected_focus
    CHECK (
      selected_focus IS NULL OR selected_focus IN (
        'migraine','anxiety','adhd','mood','sleep','medication'
      )
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ------------------------------------------------------------
-- 2) mindmap_entries: daily MindMap Score (consistency, 0..100)
-- ------------------------------------------------------------
ALTER TABLE public.mindmap_entries
  ADD COLUMN IF NOT EXISTS mindmap_score smallint;

DO $$ BEGIN
  ALTER TABLE public.mindmap_entries
    ADD CONSTRAINT chk_mindmap_entries_score_range
    CHECK (mindmap_score IS NULL OR (mindmap_score >= 0 AND mindmap_score <= 100));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

COMMIT;
