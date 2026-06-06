/**
 * Weather (Open-Meteo)
 * --------------------
 * Free, no API key. Two helpers: geocode a city name to coordinates, and
 * fetch a single day's weather snapshot. Both return null on any failure so
 * callers can treat weather as strictly best-effort/optional.
 */

export interface GeocodeResult {
  lat: number;
  lon: number;
  label: string;
}

export interface DailyWeather {
  temp_max: number | null;
  temp_min: number | null;
  precipitation: number | null;
  humidity: number | null;
  pressure: number | null;
  weather_code: number | null;
}

export type PollenLevel = "low" | "moderate" | "high" | "very_high";

export interface AirQuality {
  aqi: number | null;
  pm25: number | null;
  pollen_tree: number | null;
  pollen_grass: number | null;
  pollen_weed: number | null;
  pollen_level: PollenLevel | null;
}

function mean(nums: unknown[]): number | null {
  const valid = nums.filter((n): n is number => typeof n === "number" && !Number.isNaN(n));
  if (valid.length === 0) return null;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
}

const round = (n: number | null, dp = 0): number | null =>
  n == null ? null : Math.round(n * 10 ** dp) / 10 ** dp;

/** Map peak pollen grains/m³ to a coarse level (Open-Meteo European scale). */
function pollenLevel(peak: number | null): PollenLevel | null {
  if (peak == null) return null;
  if (peak < 20) return "low";
  if (peak < 50) return "moderate";
  if (peak < 120) return "high";
  return "very_high";
}

export async function geocodeCity(name: string): Promise<GeocodeResult | null> {
  const q = name.trim();
  if (!q) return null;
  try {
    const url = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(q)}&count=1&language=en&format=json`;
    const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!res.ok) return null;
    const data = await res.json();
    const hit = data?.results?.[0];
    if (!hit) return null;
    const label = [hit.name, hit.admin1, hit.country_code].filter(Boolean).join(", ");
    return { lat: hit.latitude, lon: hit.longitude, label };
  } catch {
    return null;
  }
}

/**
 * Fetch one day's weather. `date` is an ISO date string (YYYY-MM-DD).
 * Pressure and humidity are averaged from hourly data; temp/precip are daily.
 */
export async function fetchDailyWeather(
  lat: number,
  lon: number,
  date: string,
): Promise<DailyWeather | null> {
  try {
    const params = new URLSearchParams({
      latitude: String(lat),
      longitude: String(lon),
      start_date: date,
      end_date: date,
      daily: "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
      hourly: "surface_pressure,relative_humidity_2m",
      timezone: "auto",
    });
    const url = `https://api.open-meteo.com/v1/forecast?${params.toString()}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!res.ok) return null;
    const data = await res.json();

    const daily = data?.daily;
    const hourly = data?.hourly;

    return {
      temp_max: daily?.temperature_2m_max?.[0] ?? null,
      temp_min: daily?.temperature_2m_min?.[0] ?? null,
      precipitation: daily?.precipitation_sum?.[0] ?? null,
      weather_code: daily?.weather_code?.[0] ?? null,
      pressure: hourly?.surface_pressure ? mean(hourly.surface_pressure) : null,
      humidity: hourly?.relative_humidity_2m ? mean(hourly.relative_humidity_2m) : null,
    };
  } catch {
    return null;
  }
}

/**
 * Fetch a day's air quality + pollen from Open-Meteo's air-quality API
 * (free, no key). Tree = max(alder, birch, olive); weed = max(mugwort, ragweed).
 */
export async function fetchAirQuality(
  lat: number,
  lon: number,
  date: string,
): Promise<AirQuality | null> {
  try {
    const params = new URLSearchParams({
      latitude: String(lat),
      longitude: String(lon),
      start_date: date,
      end_date: date,
      hourly:
        "pm2_5,us_aqi,alder_pollen,birch_pollen,olive_pollen,grass_pollen,mugwort_pollen,ragweed_pollen",
      timezone: "auto",
    });
    const url = `https://air-quality-api.open-meteo.com/v1/air-quality?${params.toString()}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!res.ok) return null;
    const h = (await res.json())?.hourly;
    if (!h) return null;

    const alder = mean(h.alder_pollen ?? []);
    const birch = mean(h.birch_pollen ?? []);
    const olive = mean(h.olive_pollen ?? []);
    const grass = mean(h.grass_pollen ?? []);
    const mugwort = mean(h.mugwort_pollen ?? []);
    const ragweed = mean(h.ragweed_pollen ?? []);

    const tree = round(Math.max(alder ?? 0, birch ?? 0, olive ?? 0));
    const grassV = round(grass);
    const weed = round(Math.max(mugwort ?? 0, ragweed ?? 0));
    const peak = Math.max(tree ?? 0, grassV ?? 0, weed ?? 0);

    return {
      aqi: round(mean(h.us_aqi ?? [])),
      pm25: round(mean(h.pm2_5 ?? []), 2),
      pollen_tree: tree,
      pollen_grass: grassV,
      pollen_weed: weed,
      pollen_level: pollenLevel(h.grass_pollen || h.birch_pollen ? peak : null),
    };
  } catch {
    return null;
  }
}
