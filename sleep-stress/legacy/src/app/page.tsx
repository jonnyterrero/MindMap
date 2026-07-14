"use client";

import { useState } from "react";
import DashboardHeader from "@/components/DashboardHeader";
import DashboardMetrics from "@/components/DashboardMetrics";
import QuickLogPanel from "@/components/QuickLogPanel";
import SleepTrackingSection from "@/components/SleepTrackingSection";
import MoodStressSection from "@/components/MoodStressSection";
import HealthCorrelations from "@/components/HealthCorrelations";
import SymptomTracking from "@/components/SymptomTracking";
import AdvancedInsightsPanel from "@/components/AdvancedInsightsPanel";
import ProfileTab from "@/components/ProfileTab";

export default function Home() {
  const [activeTab, setActiveTab] = useState("dashboard");

  // Render content based on active tab
  const renderTabContent = () => {
    switch (activeTab) {
      case "dashboard":
        return (
          <>
            {/* Welcome Section */}
            <div className="mb-6 sm:mb-8">
              <h2 className="text-2xl sm:text-3xl font-bold mb-2">Welcome back!</h2>
              <p className="text-sm sm:text-base text-muted-foreground">Here's your health overview for today</p>
            </div>

            {/* Main Grid Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-6">
              {/* Left Column - Main Content */}
              <div className="lg:col-span-8 space-y-4 sm:space-y-6">
                {/* Health Metrics Cards */}
                <DashboardMetrics />

                {/* Sleep Tracking */}
                <SleepTrackingSection />

                {/* Mood & Stress */}
                <MoodStressSection />

                {/* Health Correlations */}
                <HealthCorrelations />

                {/* Symptom Tracking */}
                <SymptomTracking />
              </div>

              {/* Right Column - Sidebar */}
              <div className="lg:col-span-4 space-y-4 sm:space-y-6">
                <QuickLogPanel />
                <AdvancedInsightsPanel />
              </div>
            </div>
          </>
        );

      case "insights":
        return (
          <div className="space-y-4 sm:space-y-6">
            <div className="mb-6 sm:mb-8">
              <h2 className="text-2xl sm:text-3xl font-bold mb-2">Health Insights</h2>
              <p className="text-sm sm:text-base text-muted-foreground">Personalized insights based on your health data</p>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
              <AdvancedInsightsPanel />
              <HealthCorrelations />
            </div>
          </div>
        );

      case "trends":
        return (
          <div className="space-y-4 sm:space-y-6">
            <div className="mb-6 sm:mb-8">
              <h2 className="text-2xl sm:text-3xl font-bold mb-2">Health Trends</h2>
              <p className="text-sm sm:text-base text-muted-foreground">Visualize your health patterns over time</p>
            </div>
            <div className="grid grid-cols-1 gap-4 sm:gap-6">
              <SleepTrackingSection />
              <MoodStressSection />
              <SymptomTracking />
            </div>
          </div>
        );

      case "achievements":
        return (
          <div className="space-y-4 sm:space-y-6">
            <div className="mb-6 sm:mb-8">
              <h2 className="text-2xl sm:text-3xl font-bold mb-2">Achievements</h2>
              <p className="text-sm sm:text-base text-muted-foreground">Track your progress and earn badges</p>
            </div>
            <AdvancedInsightsPanel />
          </div>
        );

      case "profile":
        return (
          <div className="space-y-4 sm:space-y-6">
            <div className="mb-6 sm:mb-8">
              <h2 className="text-2xl sm:text-3xl font-bold mb-2">Your Profile</h2>
              <p className="text-sm sm:text-base text-muted-foreground">Manage your personal information</p>
            </div>
            <ProfileTab />
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      <DashboardHeader activeTab={activeTab} onTabChange={setActiveTab} />
      
      <main className="max-w-[1600px] mx-auto px-4 sm:px-6 py-6 sm:py-8">
        {renderTabContent()}
      </main>
    </div>
  );
}