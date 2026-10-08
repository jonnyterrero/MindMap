# Supabase advisor hardening — 2026-10-08

Closes two of the parked Supabase security-advisor WARNs from the beta
backlog (DEVELOPMENT_PLAN "Still open" + the 2026-10-07 resume note):

- **lint 0014 `extension_in_public`** — `pg_net` installed in `public`.
- **lint 0026 `pg_graphql_anon_table_exposed`** — all 46 public tables
  discoverable by the `anon` role through GraphQL/REST.

Written in cloud session on branch `claude/zealous-planck-3jyp9d`.
Project: `MindMap+_Backend` (`zunpccwjghwpiljwwjpv`). Nothing here was
applied to prod by this session — **you apply the migrations when ready.**

## What shipped (repo only)

| File | Lint | Risk | Apply |
|---|---|---|---|
| `supabase/migrations/029_anon_graphql_lockdown.sql` | 0026 | **Safe** — grant change, RLS unchanged | Any time |
| `supabase/migrations/029b_pg_net_relocate.sql` | 0014 | **Windowed** — DROP/CREATE of a cron-critical extension | Maintenance window |

## Live state this was written against (read-only verification)

- `pg_net` → registered in `public`; **all 15 of its objects already live
  in the `net` schema**, zero in `public`. `pg_net.relocatable = false`.
- `anon` holds the full table ACL (`arwdDxtm`) on every `public` table,
  re-granted to future tables by default privileges owned by `postgres`
  **and** `supabase_admin`. `anon` has `USAGE` on `public`.
- `extensions` schema exists (already home to `pgcrypto`, `uuid-ossp`,
  `pg_stat_statements`); `pg_cron` is already safely in `pg_catalog`.

## 029 — anon GraphQL lockdown (do this first; it's safe)

Revokes all table privileges from `anon` in `public`, re-grants `SELECT`
on `legal_documents` only (public ToS/privacy/disclaimer, RLS-gated to
`is_active = true`), and revokes the default privileges that would
re-expose future tables.

**Why it's safe:** the app reads user data as the `authenticated` role,
never `anon`. Pre-sign-in, the only table the client touches is
`legal_documents`. RLS already blocked anon from every row; this removes
the schema-level *discoverability* and the latent write privileges.

Apply (either path):
```
# Supabase MCP
apply_migration(project_id=zunpccwjghwpiljwwjpv,
                name="anon_graphql_lockdown",
                query=<contents of 029_anon_graphql_lockdown.sql>)
# or paste the file into the SQL Editor.
```

Verify:
```sql
-- Expect ONLY legal_documents (SELECT). Anything else means a revoke missed.
select table_name, privilege_type
from information_schema.role_table_grants
where grantee = 'anon' and table_schema = 'public'
order by 1, 2;
```
- Load `/privacy`, `/terms`, and the consent screen **signed out** — legal
  copy must still render (proves anon `legal_documents` read survived).
- Re-run `get_advisors(type=security)` → lint 0026 should fall from 46 to
  **1** (just `legal_documents`, intentional).

## 029b — pg_net relocation (maintenance window)

`pg_net` is non-relocatable, so `ALTER EXTENSION … SET SCHEMA` is rejected;
029b does `DROP EXTENSION pg_net; CREATE EXTENSION pg_net WITH SCHEMA
extensions`. The callable functions stay in the `net` schema, so
`net.http_post(...)` call sites keep working — only the extension's
registration schema moves. It drops/recreates the ephemeral `net.*` queue
tables, so run it when **no cron job is firing**.

**Lower-risk alternative:** Dashboard → Database → Extensions → `pg_net` →
disable, then re-enable into schema `extensions`. Same result,
Supabase-managed. Either path is fine — pick one.

Before: note the cron jobs that call `net.*` so you can confirm them after.
```sql
select jobid, schedule, left(command, 80) as command
from cron.job
where command ilike '%net.http_%';
```

Verify after:
```sql
-- extension now in `extensions`
select e.extname, n.nspname
from pg_extension e join pg_namespace n on n.oid = e.extnamespace
where e.extname = 'pg_net';

-- functions still in net
select proname from pg_proc p
join pg_namespace n on n.oid = p.pronamespace
where n.nspname = 'net' order by 1;
```
- Let the next report / ML-graph cron run fire (or trigger one) and confirm
  it completes — that's the real end-to-end check that `net.http_post`
  still resolves for the scheduled jobs.
- Re-run `get_advisors(type=security)` → lint 0014 clears.

## Out of scope today (noted, not touched)

Re-running the advisor surfaced these; none are part of this change:

- **lint 0027 `pg_graphql_authenticated_table_exposed`** (46 tables). The
  app legitimately reads these as the `authenticated` role via RLS —
  revoking `SELECT` from `authenticated` would break the product. Closing
  it means moving reads behind RPCs/views, which is a real redesign, not a
  grant tweak. Leave for a deliberate decision.
- **lint 0029 `authenticated_security_definer_function_executable`** (7
  provider RPCs + `rpc_consume_ai_quota` + `lookup_provider`). The provider
  RPCs are SECURITY DEFINER by design and already identify the caller via
  `auth.uid()`; `rpc_consume_ai_quota` is called by the app as
  `authenticated`. Worth a pass later to confirm each is intended.
- **`auth_leaked_password_protection`** — still off; needs Supabase Pro.
  Already parked in prior handoffs.
- **lint 0008 `rls_enabled_no_policy`** on `athlete.dose_log`,
  `athlete.sigma_cycles`, `athlete.supplements`, `athlete.training_sessions`
  — a non-MindMap `athlete` schema sharing this database. Not ours; flagging
  only so it isn't mistaken for a MindMap regression.
