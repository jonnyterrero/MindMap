# Local note — 2026-10-07 calendar day

Branch: `fix/local-calendar-day` (not merged).

Check-ins, meds, weather, body map, journal dates, and the Home streak used `toISOString()`, which is UTC. After 8pm Eastern that stored the next calendar day, so an evening check-in showed up as tomorrow and could overwrite the real next morning. Those paths now use the profile timezone (`America/New_York` when unset). Client forms use the browser's calendar day.

Voice-note work on `feat/adr-003-voice-transcript` was left untouched.
