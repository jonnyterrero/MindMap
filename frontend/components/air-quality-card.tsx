import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Wind, Flower2 } from "lucide-react";
import type { TodayWeather } from "@/app/(app)/settings/actions";

function aqiStyle(aqi: number): { label: string; pill: string } {
  if (aqi <= 50) return { label: "Good", pill: "bg-green-500/15 text-green-600 dark:text-green-400" };
  if (aqi <= 100) return { label: "Moderate", pill: "bg-amber-500/15 text-amber-600 dark:text-amber-400" };
  if (aqi <= 150) return { label: "Sensitive", pill: "bg-orange-500/15 text-orange-600 dark:text-orange-400" };
  if (aqi <= 200) return { label: "Unhealthy", pill: "bg-red-500/15 text-red-600 dark:text-red-400" };
  return { label: "Very unhealthy", pill: "bg-purple-500/15 text-purple-600 dark:text-purple-400" };
}

const POLLEN_STYLE: Record<string, string> = {
  low: "bg-green-500/15 text-green-600 dark:text-green-400",
  moderate: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  high: "bg-orange-500/15 text-orange-600 dark:text-orange-400",
  very_high: "bg-red-500/15 text-red-600 dark:text-red-400",
};

/** Renders today's AQI + pollen. Returns null when there's no air data. */
export function AirQualityCard({ weather }: { weather: TodayWeather }) {
  if (!weather || (weather.aqi == null && weather.pollen_level == null)) return null;
  const aqi = weather.aqi != null ? aqiStyle(weather.aqi) : null;

  return (
    <Card>
      <CardContent className="flex flex-wrap items-center gap-3 p-3">
        {aqi && (
          <div className="flex items-center gap-2">
            <Wind className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm">Air</span>
            <span className={cn("rounded-full px-2 py-0.5 text-xs font-medium", aqi.pill)}>
              AQI {weather.aqi} · {aqi.label}
            </span>
          </div>
        )}
        {weather.pollen_level && (
          <div className="flex items-center gap-2">
            <Flower2 className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm">Pollen</span>
            <span className={cn("rounded-full px-2 py-0.5 text-xs font-medium capitalize", POLLEN_STYLE[weather.pollen_level] ?? "")}>
              {weather.pollen_level.replace("_", " ")}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
