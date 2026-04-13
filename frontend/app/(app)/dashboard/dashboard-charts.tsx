"use client";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import { format, parseISO } from "date-fns";

type Entry = Record<string, unknown>;

function formatDate(dateStr: string) {
  return format(parseISO(dateStr), "MMM d");
}

export function DashboardCharts({ entries }: { entries: Entry[] }) {
  const moodData = entries.map((e) => ({
    date: formatDate(e.entry_date as string),
    mood: e.mood_valence as number | null,
    anxiety: e.anxiety as number | null,
    depression: e.depression as number | null,
  }));

  const sleepData = entries.map((e) => ({
    date: formatDate(e.entry_date as string),
    hours: e.sleep_minutes ? ((e.sleep_minutes as number) / 60).toFixed(1) : null,
    quality: e.sleep_quality as number | null,
  }));

  const focusData = entries.map((e) => ({
    date: formatDate(e.entry_date as string),
    focus: e.focus as number | null,
    productivity: e.productivity as number | null,
  }));

  const migraineData = entries
    .filter((e) => e.migraine === true)
    .map((e) => ({
      date: formatDate(e.entry_date as string),
      intensity: e.migraine_intensity as number | null,
    }));

  const totalEntries = entries.length;
  const migraineDays = entries.filter((e) => e.migraine === true).length;
  const avgSleep =
    entries
      .filter((e) => e.sleep_minutes != null)
      .reduce((sum, e) => sum + (e.sleep_minutes as number), 0) /
      (entries.filter((e) => e.sleep_minutes != null).length || 1) /
    60;
  const avgMood =
    entries
      .filter((e) => e.mood_valence != null)
      .reduce((sum, e) => sum + (e.mood_valence as number), 0) /
    (entries.filter((e) => e.mood_valence != null).length || 1);

  return (
    <div className="space-y-6">
      {/* Summary stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Card className="glass-card">
          <CardContent className="py-4 text-center">
            <p className="text-2xl font-bold">{totalEntries}</p>
            <p className="text-xs text-muted-foreground">Entries</p>
          </CardContent>
        </Card>
        <Card className="glass-card">
          <CardContent className="py-4 text-center">
            <p className="text-2xl font-bold">{avgSleep.toFixed(1)}h</p>
            <p className="text-xs text-muted-foreground">Avg Sleep</p>
          </CardContent>
        </Card>
        <Card className="glass-card">
          <CardContent className="py-4 text-center">
            <p className="text-2xl font-bold">
              {avgMood >= 0 ? "+" : ""}
              {avgMood.toFixed(1)}
            </p>
            <p className="text-xs text-muted-foreground">Avg Mood</p>
          </CardContent>
        </Card>
        <Card className="glass-card">
          <CardContent className="py-4 text-center">
            <p className="text-2xl font-bold">{migraineDays}</p>
            <p className="text-xs text-muted-foreground">Migraine Days</p>
          </CardContent>
        </Card>
      </div>

      {/* Mood chart */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-base">Mood & Mental Health</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={moodData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="mood"
                  stroke="hsl(270, 80%, 70%)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  name="Mood"
                  connectNulls
                />
                <Line
                  type="monotone"
                  dataKey="anxiety"
                  stroke="hsl(330, 80%, 65%)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  name="Anxiety"
                  connectNulls
                />
                <Line
                  type="monotone"
                  dataKey="depression"
                  stroke="hsl(210, 60%, 60%)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  name="Depression"
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Sleep chart */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-base">Sleep</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sleepData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar
                  dataKey="hours"
                  fill="hsl(270, 80%, 70%)"
                  radius={[4, 4, 0, 0]}
                  name="Hours"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Focus & Productivity */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-base">Focus & Productivity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={focusData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="focus"
                  stroke="hsl(270, 80%, 70%)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  name="Focus (0-10)"
                  connectNulls
                />
                <Line
                  type="monotone"
                  dataKey="productivity"
                  stroke="hsl(150, 60%, 50%)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  name="Productivity (%)"
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Migraine events */}
      {migraineData.length > 0 && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-base">Migraine Events</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={migraineData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(0,0,0,0.1)"
                  />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 10]} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar
                    dataKey="intensity"
                    fill="hsl(0, 70%, 60%)"
                    radius={[4, 4, 0, 0]}
                    name="Intensity"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
