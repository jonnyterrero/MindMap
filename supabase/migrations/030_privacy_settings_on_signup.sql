-- Provision user_privacy_settings on signup. analytics_opt_in stays false
-- (column default). No backfill: existing accounts remain opted out until
-- they use Settings or a new consent submission.
-- CREATE OR REPLACE drops SET search_path, so the pin from 011a is restated.

CREATE OR REPLACE FUNCTION public.handle_new_user()
  RETURNS trigger
  LANGUAGE plpgsql
  SECURITY DEFINER
  SET search_path = public, pg_temp
AS $$
BEGIN
  INSERT INTO public.profiles (id, display_name)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data ->> 'display_name', split_part(NEW.email, '@', 1))
  );

  INSERT INTO public.user_privacy_settings (user_id)
  VALUES (NEW.id)
  ON CONFLICT (user_id) DO NOTHING;

  RETURN NEW;
END;
$$;

ALTER FUNCTION public.handle_new_user() SET search_path = public, pg_temp;
REVOKE EXECUTE ON FUNCTION public.handle_new_user() FROM anon, authenticated;
