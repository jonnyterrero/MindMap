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

function mean(nums: number[]): number | null {
  const valid = nums.filter((n) => typeof n === "number" && !Number.isNaN(n));
  if (valid.length === 0) return null;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
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
