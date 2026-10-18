# Auth: Signup / Email-Confirmation Flow Fix

**Date:** 2026-10-18
**Area:** Supabase email-confirmation auth flow (Next.js App Router frontend)
**Related branch:** `fix/signup-email-confirmation-flow` (commit `28665a8`)

A durable reference for how MindMap's email-confirmation signup works, the bug
that broke it in production, and the fix.

---

## How the flow is supposed to work

Email confirmation is **enabled** in Supabase Auth. That means:

1. `supabase.auth.signUp()` creates the user and sends a confirmation email,
   but returns **`session: null`**. The account is not usable yet.
2. The user clicks the link → lands on `GET /auth/confirm` (see
   `frontend/app/auth/confirm/route.ts`), which calls `verifyOtp()` to exchange
   the one-time token for a session, then redirects into the app.
3. Only after that can `signInWithPassword()` succeed.

Key fact that the original code missed: **with confirmation on, `signUp()`
never returns a session.** A session only appears after the email link is
clicked.

## The bug

`frontend/app/login/actions.ts` → `signUp()` unconditionally ran
`redirect("/home")` after `signUp()`, assuming a session existed.

Resulting chain for every new user:

```
submit signup
  └─ user row created, confirmation email sent, NO auth cookie
      └─ redirect("/home")
          └─ middleware getUser() → null
              └─ redirect back to /login   (client router cache now stale)
```

### Three symptoms, one cause

| Symptom | Why |
| --- | --- |
| "Had to refresh for the landing page to load" | The `/home` → `/login` redirect chain left Next's client router cache stale. |
| "email rate limit exceeded" | Flow looked broken → users retried signup → each retry re-sent a confirmation email → Supabase's **default built-in email sender** cap (a few/hour, project-wide) blew. |
| "Invalid login credentials" after confirming | An unconfirmed account can't sign in; Supabase returns `email_not_confirmed`, historically folded into the generic "Invalid login credentials." |

### Evidence used

- `auth.users`: every recent signup had `confirmation_sent_at` set →
  confirmation is required.
- Counts at diagnosis: 8 users total, 1 unconfirmed (2026-10-08), 7 confirmed +
  signed in.
- The "email rate limit exceeded" string is specific to Supabase's default
  email service — a custom SMTP provider does not emit it.

## The fix (code)

### `signUp()` — branch on whether a session exists

```ts
const { data, error } = await supabase.auth.signUp({ email, password, options });
if (error) return { error: error.message };

// Confirmation required (or email-enumeration guard): no session returned.
// Don't redirect into the app — there's no auth cookie.
if (!data.session) {
  if (data.user) await captureServerEvent(data.user.id, AnalyticsEvent.SignedUp);
  return { confirmEmail: true, email };
}

// Confirmation disabled: a real session exists.
if (data.user) await captureServerEvent(data.user.id, AnalyticsEvent.SignedUp);
revalidatePath("/", "layout");
redirect("/home");
```

The `!data.session` branch also covers Supabase's email-enumeration guard:
signing up an address that already exists returns a user with no session and no
error. Showing the same "check your inbox" screen avoids leaking which emails
are registered.

### `signIn()` — clear message for unconfirmed accounts

```ts
if (error.code === "email_not_confirmed") {
  return {
    error: "Your email isn't confirmed yet. Check your inbox for the confirmation link — we can resend it below.",
    needsConfirmation: true,
    email,
  };
}
```

### New `resendConfirmation()` action

Wraps `supabase.auth.resend({ type: "signup", email, options })`, always
returns the same neutral success message regardless of whether the address has
a pending signup (no enumeration leak).

### UI

- **Signup page** (`frontend/app/signup/page.tsx`): on `{ confirmEmail }`,
  swaps the form for a "Confirm your email" card showing the address + a
  **Resend** button.
- **Login page** (`frontend/app/login/page.tsx`): on `{ needsConfirmation }`,
  shows a **Resend confirmation** link under the error.

## The other half of the fix (Supabase dashboard — required)

The code change stops the flow from *hammering* the cap. The rate limit itself
only goes away when the email sender is replaced. See the handoff doc section 4:

1. **Auth → Emails → SMTP Settings → Enable Custom SMTP** (Resend / SendGrid /
   SES / Postmark).
2. **Auth → Rate Limits** → raise email send limit.
3. **Auth → URL Configuration** → Site URL + Redirect URLs must include
   `https://<domain>/auth/confirm`.

## Testing checklist (after SMTP + merge)

- [ ] Fresh email → signup → see "Confirm your email" screen (no redirect loop).
- [ ] Confirmation email actually arrives (custom SMTP).
- [ ] Click link → `/auth/confirm` → lands signed in.
- [ ] Try logging in before confirming → clear message + working resend.
- [ ] Resend button delivers a second email without tripping the limit.
- [ ] Existing-email signup → same confirmation screen (no enumeration leak).
