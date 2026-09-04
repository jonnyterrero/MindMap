-- 025_insight_feedback.sql
-- "Was this helpful?" feedback on generated insights (Beta plan 1.6).
-- One row per (user, insight); re-voting updates the row. Feeds future
-- tuning of the rule engines and is required signal before trusting any
-- trained model. Idempotent: safe to re-run.

create table if not exists public.mindmap_insight_feedback (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  insight_id uuid not null references public.mindmap_insights (id) on delete cascade,
  insight_type text not null,
  helpful boolean not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, insight_id)
);

alter table public.mindmap_insight_feedback enable row level security;

drop policy if exists insight_feedback_select_own on public.mindmap_insight_feedback;
create policy insight_feedback_select_own on public.mindmap_insight_feedback
  for select using (auth.uid() = user_id);

drop policy if exists insight_feedback_insert_own on public.mindmap_insight_feedback;
create policy insight_feedback_insert_own on public.mindmap_insight_feedback
  for insert with check (auth.uid() = user_id);

drop policy if exists insight_feedback_update_own on public.mindmap_insight_feedback;
create policy insight_feedback_update_own on public.mindmap_insight_feedback
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

create index if not exists idx_insight_feedback_user
  on public.mindmap_insight_feedback (user_id, created_at desc);

comment on table public.mindmap_insight_feedback is
  'Per-user thumbs up/down on generated insights. Unique per (user, insight); re-votes update.';
