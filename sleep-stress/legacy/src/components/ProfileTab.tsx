"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { User, Save, Loader2 } from "lucide-react";
import { toast } from "sonner";

interface UserProfile {
  name: string;
  gender: string;
  height: string;
  weight: string;
  conditions: string;
}

export default function ProfileTab() {
  const [profile, setProfile] = useState<UserProfile>({
    name: "",
    gender: "",
    height: "",
    weight: "",
    conditions: "",
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [bmi, setBmi] = useState<number | null>(null);

  useEffect(() => {
    fetchProfile();
  }, []);

  useEffect(() => {
    // Calculate BMI when height and weight change
    if (profile.height && profile.weight) {
      const heightInMeters = parseFloat(profile.height) / 100;
      const weightInKg = parseFloat(profile.weight);
      if (heightInMeters > 0 && weightInKg > 0) {
        const calculatedBmi = weightInKg / (heightInMeters * heightInMeters);
        setBmi(Math.round(calculatedBmi * 10) / 10);
      }
    } else {
      setBmi(null);
    }
  }, [profile.height, profile.weight]);

  const fetchProfile = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/user-profiles");
      if (response.ok) {
        const data = await response.json();
        if (data.profiles && data.profiles.length > 0) {
          const latestProfile = data.profiles[0];
          setProfile({
            name: latestProfile.name || "",
            gender: latestProfile.gender || "",
            height: latestProfile.height_cm?.toString() || "",
            weight: latestProfile.weight_kg?.toString() || "",
            conditions: latestProfile.known_conditions?.join(", ") || "",
          });
        }
      }
    } catch (error) {
      console.error("Error fetching profile:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!profile.name.trim()) {
      toast.error("Please enter your name");
      return;
    }

    setSaving(true);
    try {
      const profileData = {
        name: profile.name,
        gender: profile.gender || null,
        height_cm: profile.height ? parseFloat(profile.height) : null,
        weight_kg: profile.weight ? parseFloat(profile.weight) : null,
        known_conditions: profile.conditions
          ? profile.conditions.split(",").map((c) => c.trim()).filter(Boolean)
          : [],
      };

      const response = await fetch("/api/user-profiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(profileData),
      });

      if (response.ok) {
        toast.success("Profile saved successfully!");
      } else {
        const error = await response.json();
        toast.error(error.error || "Failed to save profile");
      }
    } catch (error) {
      console.error("Error saving profile:", error);
      toast.error("Failed to save profile");
    } finally {
      setSaving(false);
    }
  };

  const getBmiCategory = (bmi: number) => {
    if (bmi < 18.5) return { label: "Underweight", color: "text-blue-600" };
    if (bmi < 25) return { label: "Normal weight", color: "text-green-600" };
    if (bmi < 30) return { label: "Overweight", color: "text-yellow-600" };
    return { label: "Obese", color: "text-red-600" };
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="w-full max-w-3xl mx-auto space-y-6 px-4 sm:px-0">
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10">
              <User className="w-5 h-5 sm:w-6 sm:h-6 text-primary" />
            </div>
            <div>
              <CardTitle className="text-lg sm:text-xl">Personal Profile</CardTitle>
              <CardDescription className="text-xs sm:text-sm">Manage your personal information and health details</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 sm:space-y-6">
          {/* Name */}
          <div className="space-y-2">
            <Label htmlFor="name" className="text-sm">Name *</Label>
            <Input
              id="name"
              placeholder="Enter your name"
              value={profile.name}
              onChange={(e) => setProfile({ ...profile, name: e.target.value })}
              className="text-base"
            />
          </div>

          {/* Gender */}
          <div className="space-y-2">
            <Label htmlFor="gender" className="text-sm">Gender</Label>
            <Select value={profile.gender} onValueChange={(value) => setProfile({ ...profile, gender: value })}>
              <SelectTrigger id="gender" className="text-base">
                <SelectValue placeholder="Select your gender" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="male">Male</SelectItem>
                <SelectItem value="female">Female</SelectItem>
                <SelectItem value="non-binary">Non-binary</SelectItem>
                <SelectItem value="trans-woman">Trans woman</SelectItem>
                <SelectItem value="trans-man">Trans man</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Height and Weight */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="height" className="text-sm">Height (cm)</Label>
              <Input
                id="height"
                type="number"
                inputMode="decimal"
                placeholder="170"
                value={profile.height}
                onChange={(e) => setProfile({ ...profile, height: e.target.value })}
                className="text-base"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="weight" className="text-sm">Weight (kg)</Label>
              <Input
                id="weight"
                type="number"
                inputMode="decimal"
                placeholder="70"
                value={profile.weight}
                onChange={(e) => setProfile({ ...profile, weight: e.target.value })}
                className="text-base"
              />
            </div>
          </div>

          {/* BMI Display */}
          {bmi && (
            <div className="p-4 rounded-lg bg-muted">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs sm:text-sm text-muted-foreground">Body Mass Index (BMI)</p>
                  <p className="text-xl sm:text-2xl font-bold">{bmi}</p>
                </div>
                <div className="text-right">
                  <p className={`text-xs sm:text-sm font-medium ${getBmiCategory(bmi).color}`}>
                    {getBmiCategory(bmi).label}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Known Conditions */}
          <div className="space-y-2">
            <Label htmlFor="conditions" className="text-sm">Known Conditions</Label>
            <Textarea
              id="conditions"
              placeholder="Enter any known health conditions, separated by commas (e.g., anxiety, insomnia, migraine)"
              value={profile.conditions}
              onChange={(e) => setProfile({ ...profile, conditions: e.target.value })}
              rows={4}
              className="text-base resize-none"
            />
            <p className="text-xs text-muted-foreground">
              Separate multiple conditions with commas
            </p>
          </div>

          {/* Save Button */}
          <Button onClick={handleSave} disabled={saving} className="w-full h-11 text-base">
            {saving ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                Save Profile
              </>
            )}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}