/**
 * Device notification scheduling for reminders (Beta plan 1.2).
 *
 * MindMap ships as a hosted web app inside a Capacitor shell, so local
 * notifications only exist on iOS/Android. On the plain web these helpers
 * no-op and the UI explains that reminders ring on the phone app.
 *
 * Sync model: the database (mindmap_reminders) is the source of truth;
 * after any change we cancel all pending notifications and reschedule from
 * the current reminder list. Idempotent and cheap at this scale.
 */
import { Capacitor } from "@capacitor/core";
import { LocalNotifications } from "@capacitor/local-notifications";

export type ReminderRow = {
  id: string;
  title: string;
  description: string | null;
  reminder_type: string;
  reminder_time: string; // "HH:MM:SS" (time without time zone)
  days_of_week: number[] | null; // 1=Sunday … 7=Saturday (Capacitor convention)
  is_active: boolean;
};

export function canScheduleDeviceNotifications(): boolean {
  return Capacitor.isNativePlatform();
}

/** Ask for (or confirm) notification permission. False on web or denial. */
export async function ensureNotificationPermission(): Promise<boolean> {
  if (!canScheduleDeviceNotifications()) return false;
  const status = await LocalNotifications.checkPermissions();
  if (status.display === "granted") return true;
  const req = await LocalNotifications.requestPermissions();
  return req.display === "granted";
}

/**
 * Deterministic 32-bit notification id from reminder uuid + weekday so a
 * re-sync replaces rather than duplicates. Java int range on Android.
 */
function notificationId(reminderId: string, weekday: number): number {
  let hash = weekday;
  for (let i = 0; i < reminderId.length; i++) {
    hash = (hash * 31 + reminderId.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % 2_000_000_000;
}

function defaultBody(reminderType: string): string {
  switch (reminderType) {
    case "Check-in": return "Take a minute to log how today went.";
    case "Medication": return "Time for your medication.";
    case "Journal": return "A few lines in your journal keeps the streak alive.";
    case "Routine": return "Time for your routine.";
    case "Therapy": return "Therapy reminder.";
    default: return "Reminder from MindMap.";
  }
}

/** Cancel everything pending and reschedule from the given reminders. */
export async function syncDeviceNotifications(reminders: ReminderRow[]): Promise<void> {
  if (!canScheduleDeviceNotifications()) return;

  const pending = await LocalNotifications.getPending();
  if (pending.notifications.length > 0) {
    await LocalNotifications.cancel({
      notifications: pending.notifications.map((n) => ({ id: n.id })),
    });
  }

  const notifications = [];
  for (const r of reminders) {
    if (!r.is_active) continue;
    const [hourStr, minuteStr] = r.reminder_time.split(":");
    const hour = Number(hourStr);
    const minute = Number(minuteStr);
    if (Number.isNaN(hour) || Number.isNaN(minute)) continue;

    const body = r.description?.trim() || defaultBody(r.reminder_type);
    const days = r.days_of_week?.length ? r.days_of_week : null;

    if (days) {
      // One repeating notification per selected weekday.
      for (const weekday of days) {
        notifications.push({
          id: notificationId(r.id, weekday),
          title: r.title,
          body,
          schedule: { on: { weekday, hour, minute }, allowWhileIdle: true },
        });
      }
    } else {
      // Every day at the given time.
      notifications.push({
        id: notificationId(r.id, 0),
        title: r.title,
        body,
        schedule: { on: { hour, minute }, allowWhileIdle: true },
      });
    }
  }

  if (notifications.length > 0) {
    await LocalNotifications.schedule({ notifications });
  }
}
