-- 024_ai_usage_limits.sql
-- Per-user daily quotas for AI (Anthropic) endpoints. Cost/abuse guardrail:
-- every user-initiated AI call consumes one unit from a per-endpoint daily
-- counter via an atomic SECURITY DEFINER RPC. The cron report job is exempt
-- (server-initiated, already capped by MAX_USERS_PER_RUN + skipIfExists).
-- Idempotent: safe to re-run.

create table if not exists public.mindmap_ai_usage (
  user_id uuid not null references auth.users (id) on delete cascade,
  endpoint text not null,
  usage_date date not null default ((now() at time zone 'utc'))::date,
  used integer not null default 0,
  updated_at timestamptz not null default now(),
  primary key (user_id, endpoint, usage_date)
);

alter table public.mindmap_ai_usage enable row level security;

-- Users may see their own usage (e.g. to render "X of Y left today").
drop policy if exists mindmap_ai_usage_select_own on public.mindmap_ai_usage;
create policy mindmap_ai_usage_select_own on public.mindmap_ai_usage
  for select using (auth.uid() = user_id);

-- No INSERT/UPDATE/DELETE policies on purpose: the only write path is the
-- SECURITY DEFINER function below, so clients cannot reset their own counters.

-- Atomically consume one unit of quota for the calling user.
-- Returns whether the call is allowed plus the post-call usage numbers.
create or replace function public.rpc_consume_ai_quota(
  p_endpoint text,
  p_limit integer
)
returns table (allowed boolean, used integer, quota integer)
language plpgsql
security definer
set search_path = public
as $$
declare
  v_user uuid := auth.uid();
  v_today date := ((now() at time zone 'utc'))::date;
  v_used integer;
begin
  if v_user is null then
    return query select false, 0, p_limit;
    return;
  end if;

  -- Single atomic upsert: increments only while under the limit, so
  -- concurrent requests cannot overshoot.
  insert into public.mindmap_ai_usage as u (user_id, endpoint, usage_date, used)
  values (v_user, p_endpoint, v_today, 1)
  on conflict (user_id, endpoint, usage_date)
  do update set used = u.used + 1, updated_at = now()
    where u.used < p_limit
  returning u.used into v_used;

  if v_used is null then
    -- Conflict row existed but the WHERE blocked the update: limit reached.
    select u.used into v_used
    from public.mindmap_ai_usage u
    where u.user_id = v_user and u.endpoint = p_endpoint and u.usage_date = v_today;
    return query select false, coalesce(v_used, p_limit), p_limit;
  else
    return query select true, v_used, p_limit;
  end if;
end;
$$;

revoke all on function public.rpc_consume_ai_quota(text, integer) from public;
revoke all on function public.rpc_consume_ai_quota(text, integer) from anon;
grant execute on function public.rpc_consume_ai_quota(text, integer) to authenticated;

comment on table public.mindmap_ai_usage is
  'Per-user, per-endpoint daily AI usage counters. Written only via rpc_consume_ai_quota.';
comment on function public.rpc_consume_ai_quota(text, integer) is
  'Atomically consume one unit of daily AI quota for the calling user. Returns (allowed, used, quota).';
