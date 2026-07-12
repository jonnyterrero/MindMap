"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Moon, Heart, Brain, Plus, Loader2 } from "lucide-react";
import { toast } from "sonner";

export default function QuickLogPanel() {
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    sleepStart: "",
    sleepEnd: "",
    sleepQuality: "",
    moodScore: "",
    stressScore: "",
    giFlare: "",
    skinFlare: "",
    migraine: "",
    journalEntry: "",
  });

  const resetForm = () => {
    setFormData({
      sleepStart: "",
      sleepEnd: "",
      sleepQuality: "",
      moodScore: "",
      stressScore: "",
      giFlare: "",
      skinFlare: "",
      migraine: "",
      journalEntry: "",
    });
  };

  const handleSubmit = async () => {
    // Validation
    if (!formData.sleepStart || !formData.sleepEnd) {
      toast.error("Please enter sleep start and end times");
      return;
    }
    if (!formData.sleepQuality || !formData.moodScore || !formData.stressScore) {
      toast.error("Please fill in all required fields");
      return;
    }

    const quality = parseFloat(formData.sleepQuality);
    const mood = parseFloat(formData.moodScore);
    const stress = parseFloat(formData.stressScore);

    if (quality < 1 || quality > 10 || mood < 1 || mood > 10 || stress < 1 || stress > 10) {
      toast.error("Scores must be between 1 and 10");
      return;
    }

    setLoading(true);
    try {
      const logData = {
        date: new Date().toISOString().split("T")[0],
        sleep_start_time: formData.sleepStart,
        sleep_end_time: formData.sleepEnd,
        sleep_quality_score: quality,
        mood_score: mood,
        stress_score: stress,
        gi_flare: formData.giFlare ? parseFloat(formData.giFlare) : 0,
        skin_flare: formData.skinFlare ? parseFloat(formData.skinFlare) : 0,
        migraine: formData.migraine ? parseFloat(formData.migraine) : 0,
        journal_entry: formData.journalEntry || null,
      };

      const response = await fetch("/api/health-logs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(logData),
      });

      if (response.ok) {
        toast.success("Health log saved successfully!");
        resetForm();
        // Trigger a page refresh to update stats
        window.location.reload();
      } else {
        const error = await response.json();
        toast.error(error.error || "Failed to save log");
      }
    } catch (error) {
      console.error("Error saving log:", error);
      toast.error("Failed to save log");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg sm:text-xl">
          <Plus className="w-4 h-4 sm:w-5 sm:h-5" />
          Quick Log
        </CardTitle>
        <CardDescription className="text-xs sm:text-sm">Log your daily health metrics</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Sleep Times */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Moon className="w-4 h-4 text-blue-600" />
            Sleep
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label htmlFor="sleepStart" className="text-xs">Start Time *</Label>
              <Input
                id="sleepStart"
                type="time"
                value={formData.sleepStart}
                onChange={(e) => setFormData({ ...formData, sleepStart: e.target.value })}
                className="text-base"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="sleepEnd" className="text-xs">End Time *</Label>
              <Input
                id="sleepEnd"
                type="time"
                value={formData.sleepEnd}
                onChange={(e) => setFormData({ ...formData, sleepEnd: e.target.value })}
                className="text-base"
              />
            </div>
          </div>
          <div className="space-y-1">
            <Label htmlFor="sleepQuality" className="text-xs">Quality (1-10) *</Label>
            <Input
              id="sleepQuality"
              type="number"
              inputMode="decimal"
              min="1"
              max="10"
              step="0.1"
              placeholder="7.5"
              value={formData.sleepQuality}
              onChange={(e) => setFormData({ ...formData, sleepQuality: e.target.value })}
              className="text-base"
            />
          </div>
        </div>

        {/* Mood & Stress */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Heart className="w-4 h-4 text-pink-600" />
            Mood & Stress
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label htmlFor="moodScore" className="text-xs">Mood (1-10) *</Label>
              <Input
                id="moodScore"
                type="number"
                inputMode="decimal"
                min="1"
                max="10"
                step="0.1"
                placeholder="7.0"
                value={formData.moodScore}
                onChange={(e) => setFormData({ ...formData, moodScore: e.target.value })}
                className="text-base"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="stressScore" className="text-xs">Stress (1-10) *</Label>
              <Input
                id="stressScore"
                type="number"
                inputMode="decimal"
                min="1"
                max="10"
                step="0.1"
                placeholder="5.0"
                value={formData.stressScore}
                onChange={(e) => setFormData({ ...formData, stressScore: e.target.value })}
                className="text-base"
              />
            </div>
          </div>
        </div>

        {/* Symptoms (Optional) */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Brain className="w-4 h-4 text-purple-600" />
            Symptoms (Optional)
          </div>
          <div className="grid grid-cols-3 gap-2 sm:gap-3">
            <div className="space-y-1">
              <Label htmlFor="giFlare" className="text-xs">GI Flare</Label>
              <Input
                id="giFlare"
                type="number"
                inputMode="decimal"
                min="0"
                max="10"
                step="0.1"
                placeholder="0"
                value={formData.giFlare}
                onChange={(e) => setFormData({ ...formData, giFlare: e.target.value })}
                className="text-base"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="skinFlare" className="text-xs">Skin Flare</Label>
              <Input
                id="skinFlare"
                type="number"
                inputMode="decimal"
                min="0"
                max="10"
                step="0.1"
                placeholder="0"
                value={formData.skinFlare}
                onChange={(e) => setFormData({ ...formData, skinFlare: e.target.value })}
                className="text-base"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="migraine" className="text-xs">Migraine</Label>
              <Input
                id="migraine"
                type="number"
                inputMode="decimal"
                min="0"
                max="10"
                step="0.1"
                placeholder="0"
                value={formData.migraine}
                onChange={(e) => setFormData({ ...formData, migraine: e.target.value })}
                className="text-base"
              />
            </div>
          </div>
        </div>

        {/* Journal Entry */}
        <div className="space-y-2">
          <Label htmlFor="journalEntry" className="text-sm">Journal Entry (Optional)</Label>
          <Textarea
            id="journalEntry"
            placeholder="How are you feeling today? Any notes about your day..."
            rows={3}
            value={formData.journalEntry}
            onChange={(e) => setFormData({ ...formData, journalEntry: e.target.value })}
            className="text-base resize-none"
          />
        </div>

        {/* Submit Button */}
        <Button onClick={handleSubmit} disabled={loading} className="w-full h-11 text-base">
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Saving...
            </>
          ) : (
            "Log Today's Health"
          )}
        </Button>
      </CardContent>
    </Card>
  );
}