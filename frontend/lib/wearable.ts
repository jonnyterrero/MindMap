/**
 * Wearable metric metadata. The unified metric interface is the same regardless
 * of source (Apple Health / Health Connect / Fitbit / Oura / manual entry).
 *
 * On web, metrics are entered manually (and still feed the prediction engine).
 * Native auto-sync (HealthKit / Health Connect via Capacitor plugins) is the
 * remaining native bridge — see MOBILE.md.
 */

export type WearableMetric = "hrv" | "sleep_score" | "resting_hr" | "steps" | "spo2" | "temperature";

export const METRIC_META: Record<WearableMetric, { label: string; unit: string }> = {
  hrv: { label: "HRV", unit: "ms" },
  sleep_score: { label: "Sleep score", unit: "/100" },
  resting_hr: { label: "Resting HR", unit: "bpm" },
  steps: { label: "Steps", unit: "steps" },
  spo2: { label: "SpO₂", unit: "%" },
  temperature: { label: "Body temp", unit: "°C" },
};

export const WEARABLE_METRICS = Object.keys(METRIC_META) as WearableMetric[];

export type WearableSourceType = "apple_health" | "health_connect" | "fitbit" | "oura" | "garmin" | "whoop";

export const SOURCE_META: Record<WearableSourceType, { label: string; native: boolean }> = {
  apple_health: { label: "Apple Health", native: true },
  health_connect: { label: "Health Connect", native: true },
  fitbit: { label: "Fitbit", native: false },
  oura: { label: "Oura", native: false },
  garmin: { label: "Garmin", native: false },
  whoop: { label: "Whoop", native: false },
};
