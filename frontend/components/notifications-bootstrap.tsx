"use client";

import { useEffect } from "react";
import { getReminders } from "@/app/(app)/settings/reminder-actions";
import {
  canScheduleDeviceNotifications,
  ensureNotificationPermission,
  syncDeviceNotifications,
} from "@/lib/notifications";

/**
 * On native (Capacitor) launches, bring the device notification schedule in
 * step with the user's saved reminders — covers fresh installs and reminders
 * edited on another device. No-ops entirely on the web. Renders nothing.
 */
export function NotificationsBootstrap() {
  useEffect(() => {
    if (!canScheduleDeviceNotifications()) return;
    let cancelled = false;
    (async () => {
      try {
        const reminders = await getReminders();
        if (cancelled || reminders.length === 0) return;
        const granted = await ensureNotificationPermission();
        if (granted && !cancelled) await syncDeviceNotifications(reminders);
      } catch {
        // Never let notification sync break app startup.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return null;
}
