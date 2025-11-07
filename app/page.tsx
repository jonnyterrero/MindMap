"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { InstallButton } from "@/components/install-button"
import {
  Brain,
  Heart,
  Moon,
  Smile,
  Activity,
  Calendar,
  Pill,
  BarChart3,
  MessageCircle,
  User,
  MapPin,
  Plus,
  TrendingUp,
  Clock,
  Zap,
  Trash2,
  Check,
  X,
  Plug,
  FileText,
  Copy,
  Cloud,
  AlertCircle,
  Download,
} from "lucide-react"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"

interface MoodEntry {
  id: string
  date: string
  mood: number
  anxiety: number
  sleep: number
  energy: number
  notes: string
}

interface BodySymptom {
  id: string
  bodyPart: string
  symptomType: string
  intensity: number
  duration: string
  notes: string
  timestamp: string
}

interface Routine {
  id: string
  name: string
  type: string
  startTime: string
  endTime: string
  activities: string[]
  completed: boolean
  effectiveness: number
}

interface Medication {
  id: string
  name: string
  dose: string
  frequency: string
  reminderTimes: string[]
  lastTaken: string | null
  enabled: boolean
}

interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: string
}

interface ProfileData {
  name: string
  age: number
  gender: "male" | "female" | "other" | "non-binary"
  conditions: string[]
  emergencyContact: string
  email: string
  joinDate: string
}

// Added for API Integrations
interface ApiKey {
  id: string
  name: string
  key: string
  createdAt: string
  permissions: string[]
}

interface EmergencyAlert {
  id: string
  alertType: "high_anxiety" | "suicidal_thoughts" | "severe_migraine" | "medication_overdose" | "panic_attack"
  severity: "low" | "medium" | "high" | "critical"
  triggeredBy: string
  notes: string
  status: "active" | "acknowledged" | "resolved"
  createdAt: string
}

interface WeatherData {
  id: string
  date: string
  temperatureHigh: number
  temperatureLow: number
  humidity: number
  precipitation: number
  condition: "sunny" | "cloudy" | "rainy" | "partly_cloudy"
}

const SymptomForm = ({
  bodyPart,
  onSubmit,
}: {
  bodyPart: string
  onSubmit: (bodyPart: string, symptomType: string, intensity: number, duration: string, notes: string) => void
}) => {
  const [symptomType, setSymptomType] = useState("")
  const [intensity, setIntensity] = useState(5)
  const [duration, setDuration] = useState("")
  const [notes, setNotes] = useState("")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit(bodyPart, symptomType, intensity, duration, notes)
    setSymptomType("")
    setIntensity(5)
    setDuration("")
    setNotes("")
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <Label htmlFor="symptomType">Symptom Type</Label>
        <Input
          id="symptomType"
          value={symptomType}
          onChange={(e) => setSymptomType(e.target.value)}
          placeholder="e.g., Pain, Stiffness, Tingling"
          className="glass-input"
          required
        />
      </div>

      <div>
        <Label htmlFor="intensity">Intensity (1-10)</Label>
        <Slider
          value={[intensity]}
          onValueChange={(value) => setIntensity(value[0])}
          max={10}
          min={1}
          step={1}
          className="w-full"
        />
        <div className="text-center text-sm text-muted-foreground mt-1">{intensity}/10</div>
      </div>

      <div>
        <Label htmlFor="duration">Duration</Label>
        <select
          id="duration"
          value={duration}
          onChange={(e) => setDuration(e.target.value)}
          className="w-full p-2 rounded-lg glass-input"
          required
        >
          <option value="">Select duration</option>
          <option value="< 1 hour">Less than 1 hour</option>
          <option value="1-6 hours">1-6 hours</option>
          <option value="6-24 hours">6-24 hours</option>
          <option value="1-3 days">1-3 days</option>
          <option value="> 3 days">More than 3 days</option>
        </select>
      </div>

      <div>
        <Label htmlFor="notes">Additional Notes</Label>
        <textarea
          id="notes"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Any additional details about the symptom..."
          className="w-full p-2 rounded-lg glass-input min-h-[80px] resize-none"
        />
      </div>

      <Button type="submit" className="w-full glass-button">
        Record Symptom
      </Button>
    </form>
  )
}

const MedicationForm = ({
  onSubmit,
  initialData,
}: { onSubmit: (med: Omit<Medication, "id" | "lastTaken" | "enabled">) => void; initialData?: Medication }) => {
  const [name, setName] = useState(initialData?.name || "")
  const [dose, setDose] = useState(initialData?.dose || "")
  const [frequency, setFrequency] = useState(initialData?.frequency || "")
  const [reminderTimes, setReminderTimes] = useState(initialData?.reminderTimes || ["08:00"])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({ name, dose, frequency, reminderTimes })
    setName("")
    setDose("")
    setFrequency("")
    setReminderTimes(["08:00"])
  }

  const addReminderTime = () => {
    setReminderTimes([...reminderTimes, "08:00"])
  }

  const removeReminderTime = (index: number) => {
    setReminderTimes(reminderTimes.filter((_, i) => i !== index))
  }

  const updateReminderTime = (index: number, time: string) => {
    setReminderTimes(reminderTimes.map((t, i) => (i === index ? time : t)))
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <Label htmlFor="name">Medication Name</Label>
        <Input
          id="name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g., Vitamin D"
          className="glass-input"
          required
        />
      </div>

      <div>
        <Label htmlFor="dose">Dose</Label>
        <Input
          id="dose"
          value={dose}
          onChange={(e) => setDose(e.target.value)}
          placeholder="e.g., 1000 IU"
          className="glass-input"
          required
        />
      </div>

      <div>
        <Label htmlFor="frequency">Frequency</Label>
        <Select value={frequency} onValueChange={setFrequency}>
          <SelectTrigger className="glass-input">
            <SelectValue placeholder="Select frequency" />
          </SelectTrigger>
          <SelectContent className="glass-card">
            <SelectItem value="daily">Daily</SelectItem>
            <SelectItem value="weekly">Weekly</SelectItem>
            <SelectItem value="monthly">Monthly</SelectItem>
            <SelectItem value="as needed">As Needed</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div>
        <Label>Reminder Times</Label>
        {reminderTimes.map((time, index) => (
          <div key={index} className="flex items-center space-x-2 mb-2">
            <Input
              type="time"
              value={time}
              onChange={(e) => updateReminderTime(index, e.target.value)}
              className="glass-input"
              required
            />
            <Button type="button" variant="ghost" size="sm" onClick={() => removeReminderTime(index)}>
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        ))}
        <Button type="button" variant="outline" size="sm" className="glass bg-transparent" onClick={addReminderTime}>
          <Plus className="h-4 w-4 mr-2" />
          Add Reminder
        </Button>
      </div>

      <Button type="submit" className="w-full glass-button">
        Save Medication
      </Button>
    </form>
  )
}

export default function MindTrackApp() {
  const [currentTime, setCurrentTime] = useState(new Date())
  const [selectedTab, setSelectedTab] = useState("dashboard")

  const [moodEntries, setMoodEntries] = useState<MoodEntry[]>([])
  const [bodySymptoms, setBodySymptoms] = useState<BodySymptom[]>([])
  const [routines, setRoutines] = useState<Routine[]>([])
  const [medications, setMedications] = useState<Medication[]>([])
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [currentMood, setCurrentMood] = useState(5)
  const [currentAnxiety, setCurrentAnxiety] = useState(3)
  const [currentSleep, setCurrentSleep] = useState(7.5)
  const [currentEnergy, setCurrentEnergy] = useState(8)
  const [newEntryNotes, setNewEntryNotes] = useState("")
  const [isAddingEntry, setIsAddingEntry] = useState(false)
  const [selectedBodyPart, setSelectedBodyPart] = useState<string | null>(null)
  const [chatInput, setChatInput] = useState("")
  const [bedTime, setBedTime] = useState("22:00")
  const [wakeTime, setWakeTime] = useState("07:00")
  const [sleepQuality, setSleepQuality] = useState(7)
  const [sleepNotes, setSleepNotes] = useState("")
  const [profileData, setProfileData] = useState<ProfileData>({
    name: "John Doe",
    age: 25,
    gender: "male",
    conditions: ["anxiety", "insomnia"],
    emergencyContact: "+1-555-0123",
    email: "john.doe@example.com",
    joinDate: "2024-01-01",
  })
  const [isEditingProfile, setIsEditingProfile] = useState(false)
  const [tempProfileData, setTempProfileData] = useState<ProfileData>(profileData)

  const [isAddingRoutine, setIsAddingRoutine] = useState(false)
  const [newRoutineName, setNewRoutineName] = useState("")
  const [newRoutineType, setNewRoutineType] = useState("morning")
  const [newRoutineStartTime, setNewRoutineStartTime] = useState("07:00")
  const [newRoutineEndTime, setNewRoutineEndTime] = useState("08:00")
  const [newRoutineActivities, setNewRoutineActivities] = useState("")

  const [newCondition, setNewCondition] = useState("")

  // Added for API Integrations
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([])
  
  // Emergency Alerts
  const [emergencyAlerts, setEmergencyAlerts] = useState<EmergencyAlert[]>([])
  
  // Weather Data
  const [weatherData, setWeatherData] = useState<WeatherData[]>([])

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    loadDemoData()
    return () => clearInterval(timer)
  }, [])

  const loadDemoData = () => {
    const demoEntries: MoodEntry[] = [
      { id: "1", date: "2024-01-15", mood: 8, anxiety: 2, sleep: 8.0, energy: 9, notes: "Great day!" },
      { id: "2", date: "2024-01-14", mood: 6, anxiety: 4, sleep: 7.0, energy: 6, notes: "Feeling okay" },
      { id: "3", date: "2024-01-13", mood: 7, anxiety: 3, sleep: 7.5, energy: 7, notes: "Good sleep" },
    ]

    const demoRoutines: Routine[] = [
      {
        id: "1",
        name: "Morning Routine",
        type: "morning",
        startTime: "08:00",
        endTime: "09:00",
        activities: ["Brush teeth", "Take medication", "Exercise"],
        completed: false,
        effectiveness: 8,
      },
      {
        id: "2",
        name: "Evening Routine",
        type: "evening",
        startTime: "20:00",
        endTime: "21:00",
        activities: ["Read", "Meditation", "Prepare for bed"],
        completed: true,
        effectiveness: 7,
      },
    ]

    const demoMedications: Medication[] = [
      {
        id: "1",
        name: "Vitamin D",
        dose: "1000 IU",
        frequency: "daily",
        reminderTimes: ["08:00"],
        lastTaken: null,
        enabled: true,
      },
      {
        id: "2",
        name: "Omega-3",
        dose: "1000mg",
        frequency: "daily",
        reminderTimes: ["20:00"],
        lastTaken: "2024-01-15 20:00",
        enabled: true,
      },
    ]

    setMoodEntries(demoEntries)
    setRoutines(demoRoutines)
    setMedications(demoMedications)
    
    // Demo emergency alerts
    const demoAlerts: EmergencyAlert[] = [
      {
        id: "1",
        alertType: "high_anxiety",
        severity: "medium",
        triggeredBy: "Work stress",
        notes: "Feeling overwhelmed with deadlines",
        status: "resolved",
        createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
      },
      {
        id: "2",
        alertType: "severe_migraine",
        severity: "high",
        triggeredBy: "Weather change",
        notes: "Severe headache after weather shift",
        status: "active",
        createdAt: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
      },
    ]
    
    // Demo weather data
    const demoWeather: WeatherData[] = []
    for (let i = 0; i < 30; i++) {
      const date = new Date()
      date.setDate(date.getDate() - (29 - i))
      demoWeather.push({
        id: i.toString(),
        date: date.toISOString().split("T")[0],
        temperatureHigh: 20 + Math.random() * 10 - 5,
        temperatureLow: 10 + Math.random() * 10 - 5,
        humidity: 30 + Math.random() * 60,
        precipitation: Math.random() > 0.7 ? Math.random() * 20 : 0,
        condition: ["sunny", "cloudy", "rainy", "partly_cloudy"][Math.floor(Math.random() * 4)] as WeatherData["condition"],
      })
    }
    
    setEmergencyAlerts(demoAlerts)
    setWeatherData(demoWeather)
  }

  const toggleRoutineCompletion = (routineId: string) => {
    setRoutines(
      routines.map((routine) => (routine.id === routineId ? { ...routine, completed: !routine.completed } : routine)),
    )
  }

  const takeMedication = (medicationId: string) => {
    setMedications(
      medications.map((medication) =>
        medication.id === medicationId ? { ...medication, lastTaken: new Date().toISOString() } : medication,
      ),
    )
  }

  const addRoutine = () => {
    if (!newRoutineName.trim()) return

    const newRoutine: Routine = {
      id: Date.now().toString(),
      name: newRoutineName,
      type: newRoutineType as "morning" | "evening" | "daily",
      startTime: newRoutineStartTime,
      endTime: newRoutineEndTime,
      activities: newRoutineActivities
        .split(",")
        .map((a) => a.trim())
        .filter((a) => a),
      completed: false,
      effectiveness: 7,
    }

    setRoutines([...routines, newRoutine])
    setNewRoutineName("")
    setNewRoutineActivities("")
    setIsAddingRoutine(false)
  }

  const addCondition = () => {
    if (!newCondition.trim()) return

    setTempProfileData({
      ...tempProfileData,
      conditions: [...tempProfileData.conditions, newCondition.toLowerCase()],
    })
    setNewCondition("")
  }

  const removeCondition = (condition: string) => {
    setTempProfileData({
      ...tempProfileData,
      conditions: tempProfileData.conditions.filter((c) => c !== condition),
    })
  }

  const saveProfile = () => {
    setProfileData(tempProfileData)
    setIsEditingProfile(false)
  }

  const cancelProfileEdit = () => {
    setTempProfileData(profileData)
    setIsEditingProfile(false)
  }

  const addNewRoutine = () => {
    const newRoutine: Routine = {
      id: Date.now().toString(),
      name: newRoutineName,
      type: newRoutineType,
      startTime: newRoutineStartTime,
      endTime: newRoutineEndTime,
      activities: newRoutineActivities
        .split(",")
        .map((a) => a.trim())
        .filter(Boolean),
      completed: false,
      effectiveness: 5,
    }
    setRoutines([...routines, newRoutine])
    setNewRoutineName("")
    setNewRoutineType("morning")
    setNewRoutineStartTime("07:00")
    setNewRoutineEndTime("08:00")
    setNewRoutineActivities("")
    setIsAddingRoutine(false)
  }

  const addConditionOld = () => {
    if (newCondition.trim() && !tempProfileData.conditions.includes(newCondition.trim())) {
      setTempProfileData({
        ...tempProfileData,
        conditions: [...tempProfileData.conditions, newCondition.trim()],
      })
      setNewCondition("")
    }
  }

  const removeConditionOld = (conditionToRemove: string) => {
    setTempProfileData({
      ...tempProfileData,
      conditions: tempProfileData.conditions.filter((c) => c !== conditionToRemove),
    })
  }

  const addMoodEntry = () => {
    const newEntry: MoodEntry = {
      id: Date.now().toString(),
      date: new Date().toISOString().split("T")[0],
      mood: currentMood,
      anxiety: currentAnxiety,
      sleep: currentSleep,
      energy: currentEnergy,
      notes: newEntryNotes,
    }
    setMoodEntries([newEntry, ...moodEntries])
    setNewEntryNotes("")
    setIsAddingEntry(false)
  }

  const addSleepEntry = () => {
    // Calculate sleep duration
    const bedTimeDate = new Date(`2000-01-01T${bedTime}:00`)
    const wakeTimeDate = new Date(`2000-01-01T${wakeTime}:00`)

    // Handle overnight sleep (bedtime after midnight)
    if (wakeTimeDate < bedTimeDate) {
      wakeTimeDate.setDate(wakeTimeDate.getDate() + 1)
    }

    const sleepDuration = (wakeTimeDate.getTime() - bedTimeDate.getTime()) / (1000 * 60 * 60)

    const newEntry: MoodEntry = {
      id: Date.now().toString(),
      date: new Date().toISOString().split("T")[0],
      mood: currentMood,
      anxiety: currentAnxiety,
      sleep: Math.round(sleepDuration * 2) / 2, // Round to nearest 0.5
      energy: currentEnergy,
      notes: sleepNotes || `Bedtime: ${bedTime}, Wake: ${wakeTime}, Quality: ${sleepQuality}/10`,
    }
    setMoodEntries([newEntry, ...moodEntries])
    setSleepNotes("")
  }

  const addBodySymptom = (
    bodyPart: string,
    symptomType: string,
    intensity: number,
    duration: string,
    notes: string,
  ) => {
    const newSymptom: BodySymptom = {
      id: Date.now().toString(),
      bodyPart,
      symptomType,
      intensity,
      duration,
      notes,
      timestamp: new Date().toISOString(),
    }
    setBodySymptoms([newSymptom, ...bodySymptoms])
  }

  const toggleRoutineCompletionOld = (routineId: string) => {
    setRoutines(
      routines.map((routine) => (routine.id === routineId ? { ...routine, completed: !routine.completed } : routine)),
    )
  }

  const takeMedicationOld = (medicationId: string) => {
    setMedications(
      medications.map((med) => (med.id === medicationId ? { ...med, lastTaken: new Date().toISOString() } : med)),
    )
  }

  const sendChatMessage = () => {
    if (!chatInput.trim()) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: chatInput,
      timestamp: new Date().toISOString(),
    }

    setChatMessages([...chatMessages, userMessage])
    setChatInput("")

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: generateAIResponse(chatInput),
        timestamp: new Date().toISOString(),
      }
      setChatMessages((prev) => [...prev, aiResponse])
    }, 1000)
  }

  const generateAIResponse = (input: string): string => {
    const lowerInput = input.toLowerCase()
    if (lowerInput.includes("mood")) {
      return "I notice you're asking about mood. Regular mood tracking can help identify patterns and triggers. Your recent mood average is looking good! Consider noting what activities or events might be affecting your mood positively."
    } else if (lowerInput.includes("sleep")) {
      return "Sleep is crucial for mental health. Aim for 7-9 hours per night. Your sleep data shows some good consistency. Try establishing a consistent bedtime routine and avoid screens before bed."
    } else if (lowerInput.includes("anxiety") || lowerInput.includes("stress")) {
      return "Managing anxiety is important for overall wellbeing. Consider techniques like deep breathing, meditation, or regular exercise. Your anxiety levels seem manageable - keep tracking to identify patterns."
    } else if (lowerInput.includes("medication")) {
      return "Medication adherence is key for effectiveness. I see you have some medications set up. Remember to take them as prescribed and set up reminders if needed."
    } else {
      return "Thank you for reaching out! I'm here to help with your mental health journey. Feel free to ask about mood tracking, sleep patterns, anxiety management, or any wellness concerns you have."
    }
  }

  const updateProfile = () => {
    setProfileData(tempProfileData)
    setIsEditingProfile(false)
  }

  const cancelProfileEditOld = () => {
    setTempProfileData(profileData)
    setIsEditingProfile(false)
  }

  // Demo data for current stats
  const todayStats = {
    mood: currentMood,
    anxiety: currentAnxiety,
    sleep: currentSleep,
    energy: currentEnergy,
  }

  return (
    <div className="min-h-screen p-4 space-y-6">
      {/* Header */}
      <div className="glass-card p-6 animate-float">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-3 rounded-full bg-gradient-to-r from-pink-400/20 to-purple-400/20 backdrop-blur-sm">
              <Brain className="h-8 w-8 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
                MindTrack
              </h1>
              <p className="text-sm text-muted-foreground">
                {currentTime.toLocaleDateString("en-US", {
                  weekday: "long",
                  year: "numeric",
                  month: "long",
                  day: "numeric",
                })}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <InstallButton />
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm" className="glass bg-transparent">
                  <User className="h-4 w-4 mr-2" />
                  Profile
                </Button>
              </DialogTrigger>
              <DialogContent className="glass-card">
                <DialogHeader>
                  <DialogTitle>Profile Settings</DialogTitle>
                  <DialogDescription>Manage your personal information and preferences</DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Name</label>
                    <Input
                      value={profileData.name}
                      onChange={(e) => setProfileData({ ...profileData, name: e.target.value })}
                      className="glass"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Age</label>
                    <Input
                      type="number"
                      value={profileData.age}
                      onChange={(e) => setProfileData({ ...profileData, age: Number.parseInt(e.target.value) })}
                      className="glass"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Emergency Contact</label>
                    <Input
                      value={profileData.emergencyContact}
                      onChange={(e) => setProfileData({ ...profileData, emergencyContact: e.target.value })}
                      className="glass"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Known Conditions</label>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {profileData.conditions.map((condition, index) => (
                        <Badge key={index} variant="outline" className="glass">
                          {condition}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="w-full">
        <TabsList className="glass-card flex w-full overflow-x-auto scrollbar-hide p-2 gap-2 justify-start">
          <TabsTrigger value="dashboard" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <BarChart3 className="h-4 w-4 mr-1" />
            Dashboard
          </TabsTrigger>
          <TabsTrigger value="mood" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <Smile className="h-4 w-4 mr-1" />
            Mood
          </TabsTrigger>
          <TabsTrigger value="body" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <MapPin className="h-4 w-4 mr-1" />
            Body Map
          </TabsTrigger>
          <TabsTrigger value="routines" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <Clock className="h-4 w-4 mr-1" />
            Routines
          </TabsTrigger>
          <TabsTrigger value="medications" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <Pill className="h-4 w-4 mr-1" />
            Meds
          </TabsTrigger>
          <TabsTrigger value="sleep" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <Moon className="h-4 w-4 mr-1" />
            Sleep
          </TabsTrigger>
          <TabsTrigger value="calendar" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <Calendar className="h-4 w-4 mr-1" />
            Calendar
          </TabsTrigger>
          <TabsTrigger value="analytics" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <TrendingUp className="h-4 w-4 mr-1" />
            Analytics
          </TabsTrigger>
          <TabsTrigger value="profile" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <User className="h-4 w-4 mr-1" />
            Profile
          </TabsTrigger>
          <TabsTrigger value="chat" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <MessageCircle className="h-4 w-4 mr-1" />
            Chat
          </TabsTrigger>
          <TabsTrigger value="integrations" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <Plug className="h-4 w-4 mr-1" />
            Integrations
          </TabsTrigger>
          <TabsTrigger value="emergency" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <AlertCircle className="h-4 w-4 mr-1" />
            Emergency
          </TabsTrigger>
          <TabsTrigger value="weather" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <Cloud className="h-4 w-4 mr-1" />
            Weather
          </TabsTrigger>
          <TabsTrigger value="export" className="text-xs whitespace-nowrap min-w-fit px-3 py-2 flex-shrink-0">
            <Download className="h-4 w-4 mr-1" />
            Export
          </TabsTrigger>
        </TabsList>

        {/* Dashboard Tab */}
        <TabsContent value="dashboard" className="space-y-6">
          {/* Quick Stats */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="glass-card animate-pulse-slow">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center">
                  <Smile className="h-4 w-4 mr-2 text-primary" />
                  Mood
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-primary">{todayStats.mood}/10</div>
                <Progress value={todayStats.mood * 10} className="mt-2" />
              </CardContent>
            </Card>

            <Card className="glass-card animate-pulse-slow" style={{ animationDelay: "0.5s" }}>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center">
                  <Zap className="h-4 w-4 mr-2 text-secondary" />
                  Anxiety
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-secondary">{todayStats.anxiety}/10</div>
                <Progress value={todayStats.anxiety * 10} className="mt-2" />
              </CardContent>
            </Card>

            <Card className="glass-card animate-pulse-slow" style={{ animationDelay: "1s" }}>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center">
                  <Moon className="h-4 w-4 mr-2 text-primary" />
                  Sleep
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-primary">{todayStats.sleep}h</div>
                <Progress value={(todayStats.sleep / 10) * 100} className="mt-2" />
              </CardContent>
            </Card>

            <Card className="glass-card animate-pulse-slow" style={{ animationDelay: "1.5s" }}>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center">
                  <Activity className="h-4 w-4 mr-2 text-secondary" />
                  Energy
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-secondary">{todayStats.energy}/10</div>
                <Progress value={todayStats.energy * 10} className="mt-2" />
              </CardContent>
            </Card>
          </div>

          {/* Recent Entries */}
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Calendar className="h-5 w-5 mr-2" />
                Recent Entries
              </CardTitle>
              <CardDescription>Your latest mood and wellness tracking</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {moodEntries.slice(0, 5).map((entry, index) => (
                  <div key={entry.id} className="flex items-center justify-between p-3 glass rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="text-sm font-medium">{entry.date}</div>
                      <Badge variant="outline" className="glass">
                        Mood: {entry.mood}/10
                      </Badge>
                      <Badge variant="outline" className="glass">
                        Anxiety: {entry.anxiety}/10
                      </Badge>
                      <Badge variant="outline" className="glass">
                        Sleep: {entry.sleep}h
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <Dialog>
              <DialogTrigger asChild>
                <Button className="glass-strong h-20 flex-col space-y-2 bg-transparent" variant="outline">
                  <Plus className="h-6 w-6" />
                  <span className="text-sm">Quick Entry</span>
                </Button>
              </DialogTrigger>
              <DialogContent className="glass-card">
                <DialogHeader>
                  <DialogTitle>Quick Mood Entry</DialogTitle>
                  <DialogDescription>Log your current mood and feelings</DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Mood (1-10)</label>
                    <div className="grid grid-cols-5 gap-2 mt-2">
                      {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((mood) => (
                        <Button
                          key={mood}
                          variant={mood === currentMood ? "default" : "outline"}
                          className={`glass h-10 ${mood === currentMood ? "bg-primary text-primary-foreground" : ""}`}
                          size="sm"
                          onClick={() => setCurrentMood(mood)}
                        >
                          {mood}
                        </Button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium">Anxiety (0-10)</label>
                    <div className="grid grid-cols-6 gap-2 mt-2">
                      {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((level) => (
                        <Button
                          key={level}
                          variant={level === currentAnxiety ? "default" : "outline"}
                          className={`glass h-8 text-xs ${level === currentAnxiety ? "bg-secondary text-secondary-foreground" : ""}`}
                          size="sm"
                          onClick={() => setCurrentAnxiety(level)}
                        >
                          {level}
                        </Button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium">Quick Notes</label>
                    <Textarea
                      value={newEntryNotes}
                      onChange={(e) => setNewEntryNotes(e.target.value)}
                      placeholder="How are you feeling right now?"
                      className="glass mt-2"
                      rows={2}
                    />
                  </div>

                  <Button onClick={addMoodEntry} className="w-full glass-strong">
                    <Plus className="h-4 w-4 mr-2" />
                    Save Quick Entry
                  </Button>
                </div>
              </DialogContent>
            </Dialog>

            <Button
              className="glass-strong h-20 flex-col space-y-2 bg-transparent"
              variant="outline"
              onClick={() => setSelectedTab("medications")}
            >
              <Pill className="h-6 w-6" />
              <span className="text-sm">Take Medication</span>
            </Button>
            <Button
              className="glass-strong h-20 flex-col space-y-2 bg-transparent"
              variant="outline"
              onClick={() => setSelectedTab("body")}
            >
              <Heart className="h-6 w-6" />
              <span className="text-sm">Check Symptoms</span>
            </Button>
            <Button
              className="glass-strong h-20 flex-col space-y-2 bg-transparent"
              variant="outline"
              onClick={() => setSelectedTab("sleep")}
            >
              <Moon className="h-6 w-6" />
              <span className="text-sm">Log Sleep</span>
            </Button>
            <Button
              className="glass-strong h-20 flex-col space-y-2 bg-transparent"
              variant="outline"
              onClick={() => setSelectedTab("chat")}
            >
              <MessageCircle className="h-6 w-6" />
              <span className="text-sm">Chat Assistant</span>
            </Button>
          </div>
        </TabsContent>

        <TabsContent value="mood" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Smile className="h-5 w-5 mr-2" />
                Mood Tracking
              </CardTitle>
              <CardDescription>Track your daily mood and emotional state</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="text-center">
                <p className="text-sm text-muted-foreground mb-4">How are you feeling today?</p>
                <div className="grid grid-cols-5 gap-2">
                  {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((mood) => (
                    <Button
                      key={mood}
                      variant={mood === currentMood ? "default" : "outline"}
                      className={`glass h-12 ${mood === currentMood ? "bg-primary text-primary-foreground" : ""}`}
                      onClick={() => setCurrentMood(mood)}
                    >
                      {mood}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Anxiety Level</label>
                  <div className="grid grid-cols-6 gap-2 mt-2">
                    {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((level) => (
                      <Button
                        key={level}
                        variant={level === currentAnxiety ? "default" : "outline"}
                        className={`glass h-10 ${level === currentAnxiety ? "bg-secondary text-secondary-foreground" : ""}`}
                        size="sm"
                        onClick={() => setCurrentAnxiety(level)}
                      >
                        {level}
                      </Button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium">Sleep Hours</label>
                  <Input
                    type="number"
                    step="0.5"
                    min="0"
                    max="12"
                    value={currentSleep}
                    onChange={(e) => setCurrentSleep(Number.parseFloat(e.target.value))}
                    className="glass mt-2"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium">Energy Level</label>
                  <div className="grid grid-cols-5 gap-2 mt-2">
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((level) => (
                      <Button
                        key={level}
                        variant={level === currentEnergy ? "default" : "outline"}
                        className={`glass h-10 ${level === currentEnergy ? "bg-primary text-primary-foreground" : ""}`}
                        size="sm"
                        onClick={() => setCurrentEnergy(level)}
                      >
                        {level}
                      </Button>
                    ))}
                  </div>
                </div>

                <div>
                  <Button
                    variant="outline"
                    onClick={() => setIsAddingEntry(!isAddingEntry)}
                    className="w-full glass mb-2"
                  >
                    {isAddingEntry ? "Hide Notes" : "Add Notes (Optional)"}
                  </Button>

                  {isAddingEntry && (
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Notes about your day</label>
                      <Textarea
                        value={newEntryNotes}
                        onChange={(e) => setNewEntryNotes(e.target.value)}
                        placeholder="How was your day? What affected your mood? Any triggers or positive events?"
                        className="glass"
                        rows={3}
                      />
                    </div>
                  )}
                </div>

                <Button onClick={addMoodEntry} className="w-full glass-strong">
                  <Plus className="h-4 w-4 mr-2" />
                  Save Entry
                </Button>
              </div>
            </CardContent>
          </Card>

          {moodEntries.length > 0 && (
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Recent Entries</CardTitle>
                <CardDescription>Your latest mood tracking entries</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {moodEntries.slice(0, 5).map((entry) => (
                    <div key={entry.id} className="p-4 glass rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-sm font-medium">{entry.date}</div>
                        <div className="flex space-x-2">
                          <Badge variant="outline" className="glass text-xs">
                            Mood: {entry.mood}/10
                          </Badge>
                          <Badge variant="outline" className="glass text-xs">
                            Anxiety: {entry.anxiety}/10
                          </Badge>
                          <Badge variant="outline" className="glass text-xs">
                            Sleep: {entry.sleep}h
                          </Badge>
                          <Badge variant="outline" className="glass text-xs">
                            Energy: {entry.energy}/10
                          </Badge>
                        </div>
                      </div>
                      {entry.notes && (
                        <div className="text-sm text-muted-foreground mt-2 p-2 glass rounded">{entry.notes}</div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="body" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <MapPin className="h-5 w-5 mr-2" />
                Body Symptom Map
              </CardTitle>
              <CardDescription>Track physical symptoms and pain locations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                {[
                  { name: "Head", icon: "🧠" },
                  { name: "Neck", icon: "👔" },
                  { name: "Chest", icon: "🫁" },
                  { name: "Stomach", icon: "🤰" },
                  { name: "Back", icon: "🫂" },
                  { name: "Arms", icon: "💪" },
                  { name: "Legs", icon: "🦵" },
                  { name: "Feet", icon: "🦶" },
                ].map((bodyPart) => (
                  <Dialog key={bodyPart.name}>
                    <DialogTrigger asChild>
                      <Button
                        variant="outline"
                        className="glass-strong h-20 flex-col space-y-2 bg-transparent"
                        onClick={() => setSelectedBodyPart(bodyPart.name)}
                      >
                        <span className="text-2xl">{bodyPart.icon}</span>
                        <span className="text-sm">{bodyPart.name}</span>
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="glass-card">
                      <DialogHeader>
                        <DialogTitle>Track {bodyPart.name} Symptoms</DialogTitle>
                        <DialogDescription>Record any symptoms or pain in this area</DialogDescription>
                      </DialogHeader>
                      <SymptomForm bodyPart={bodyPart.name} onSubmit={addBodySymptom} />
                    </DialogContent>
                  </Dialog>
                ))}
              </div>
            </CardContent>
          </Card>

          {bodySymptoms.length > 0 && (
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Current Symptoms</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {bodySymptoms.slice(0, 5).map((symptom) => (
                    <div key={symptom.id} className="p-3 glass rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">
                            {symptom.bodyPart} - {symptom.symptomType}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            Intensity: {symptom.intensity}/10 • Duration: {symptom.duration}
                          </div>
                          {symptom.notes && <div className="text-sm text-muted-foreground mt-1">{symptom.notes}</div>}
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setBodySymptoms(bodySymptoms.filter((s) => s.id !== symptom.id))}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="routines" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center">
                  <Clock className="h-5 w-5 mr-2" />
                  Daily Routines
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setIsAddingRoutine(!isAddingRoutine)}
                  className="glass"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  {isAddingRoutine ? "Cancel" : "Add Routine"}
                </Button>
              </CardTitle>
              <CardDescription>Manage your wellness routines</CardDescription>
            </CardHeader>
            <CardContent>
              {isAddingRoutine && (
                <Card className="glass mb-6">
                  <CardHeader>
                    <CardTitle className="text-lg">Create New Routine</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                      <div>
                        <label className="text-sm font-medium">Routine Name</label>
                        <Input
                          value={newRoutineName}
                          onChange={(e) => setNewRoutineName(e.target.value)}
                          placeholder="Morning meditation"
                          className="glass mt-1"
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium">Type</label>
                        <select
                          value={newRoutineType}
                          onChange={(e) => setNewRoutineType(e.target.value)}
                          className="w-full mt-1 p-2 glass rounded-md border border-border bg-background"
                        >
                          <option value="morning">Morning</option>
                          <option value="evening">Evening</option>
                          <option value="exercise">Exercise</option>
                          <option value="mindfulness">Mindfulness</option>
                          <option value="self-care">Self-care</option>
                        </select>
                      </div>
                      <div>
                        <label className="text-sm font-medium">Start Time</label>
                        <Input
                          type="time"
                          value={newRoutineStartTime}
                          onChange={(e) => setNewRoutineStartTime(e.target.value)}
                          className="glass mt-1"
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium">End Time</label>
                        <Input
                          type="time"
                          value={newRoutineEndTime}
                          onChange={(e) => setNewRoutineEndTime(e.target.value)}
                          className="glass mt-1"
                        />
                      </div>
                    </div>
                    <div>
                      <label className="text-sm font-medium">Activities (comma-separated)</label>
                      <Input
                        value={newRoutineActivities}
                        onChange={(e) => setNewRoutineActivities(e.target.value)}
                        placeholder="meditation, breathing exercises, journaling"
                        className="glass mt-1"
                      />
                    </div>
                    <Button onClick={addRoutine} className="w-full glass-strong">
                      <Plus className="h-4 w-4 mr-2" />
                      Create Routine
                    </Button>
                  </CardContent>
                </Card>
              )}

              <div className="space-y-4">
                {routines.map((routine) => (
                  <div key={routine.id} className="p-4 glass rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <h3 className="font-medium">{routine.name}</h3>
                          <Badge variant={routine.completed ? "default" : "outline"} className="glass">
                            {routine.type}
                          </Badge>
                        </div>
                        <div className="text-sm text-muted-foreground mt-1">
                          {routine.startTime} - {routine.endTime}
                        </div>
                        <div className="text-sm text-muted-foreground mt-2">
                          Activities: {routine.activities.join(", ")}
                        </div>
                        <div className="text-sm text-muted-foreground">Effectiveness: {routine.effectiveness}/10</div>
                      </div>
                      <Button
                        variant={routine.completed ? "default" : "outline"}
                        size="sm"
                        onClick={() => toggleRoutineCompletion(routine.id)}
                        className="glass"
                      >
                        {routine.completed ? <Check className="h-4 w-4" /> : <Clock className="h-4 w-4" />}
                      </Button>
                    </div>
                  </div>
                ))}
                {routines.length === 0 && !isAddingRoutine && (
                  <div className="text-center py-8 text-muted-foreground">
                    <Clock className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No routines yet. Create your first routine to get started!</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="medications" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center">
                  <Pill className="h-5 w-5 mr-2" />
                  Medication Management
                </div>
                <Dialog>
                  <DialogTrigger asChild>
                    <Button variant="outline" size="sm" className="glass bg-transparent">
                      <Plus className="h-4 w-4 mr-2" />
                      Add Medication
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="glass-card">
                    <DialogHeader>
                      <DialogTitle>Add New Medication</DialogTitle>
                      <DialogDescription>Set up a new medication with reminders</DialogDescription>
                    </DialogHeader>
                    <MedicationForm
                      onSubmit={(med) =>
                        setMedications([
                          ...medications,
                          { ...med, id: Date.now().toString(), lastTaken: null, enabled: true },
                        ])
                      }
                    />
                  </DialogContent>
                </Dialog>
              </CardTitle>
              <CardDescription>Track your medications, set reminders, and monitor adherence</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {medications.map((medication) => (
                  <div key={medication.id} className="p-4 glass rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <h3 className="font-medium">{medication.name}</h3>
                          <Badge variant="outline" className="glass">
                            {medication.dose}
                          </Badge>
                          <Badge variant={medication.enabled ? "default" : "secondary"} className="glass text-xs">
                            {medication.enabled ? "Active" : "Paused"}
                          </Badge>
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {medication.frequency} at {medication.reminderTimes.join(", ")}
                        </div>
                        {medication.lastTaken && (
                          <div className="text-sm text-muted-foreground">
                            Last taken: {new Date(medication.lastTaken).toLocaleString()}
                          </div>
                        )}

                        <div className="mt-2">
                          <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                            <span>Weekly Adherence</span>
                            <span>85%</span>
                          </div>
                          <Progress value={85} className="h-2" />
                        </div>
                      </div>
                      <div className="flex flex-col space-y-2 ml-4">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => takeMedication(medication.id)}
                          className="glass"
                          disabled={
                            medication.lastTaken &&
                            new Date().toDateString() === new Date(medication.lastTaken).toDateString()
                          }
                        >
                          <Check className="h-4 w-4 mr-1" />
                          {medication.lastTaken &&
                          new Date().toDateString() === new Date(medication.lastTaken).toDateString()
                            ? "Taken Today"
                            : "Take Now"}
                        </Button>

                        <Dialog>
                          <DialogTrigger asChild>
                            <Button variant="ghost" size="sm" className="glass text-xs">
                              Edit
                            </Button>
                          </DialogTrigger>
                          <DialogContent className="glass-card">
                            <DialogHeader>
                              <DialogTitle>Edit Medication</DialogTitle>
                              <DialogDescription>Update medication details and settings</DialogDescription>
                            </DialogHeader>
                            <MedicationForm
                              initialData={medication}
                              onSubmit={(updatedMed) => {
                                setMedications(
                                  medications.map((med) =>
                                    med.id === medication.id
                                      ? {
                                          ...updatedMed,
                                          id: medication.id,
                                          lastTaken: medication.lastTaken,
                                          enabled: medication.enabled,
                                        }
                                      : med,
                                  ),
                                )
                              }}
                            />
                          </DialogContent>
                        </Dialog>
                      </div>
                    </div>
                  </div>
                ))}

                {medications.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    <Pill className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No medications added yet</p>
                    <p className="text-sm mt-2">Add your first medication to start tracking</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {medications.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="text-lg">Today's Schedule</CardTitle>
                  <CardDescription>Upcoming medication reminders</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {medications
                      .flatMap((med) =>
                        med.reminderTimes.map((time) => ({
                          ...med,
                          reminderTime: time,
                          taken: med.lastTaken && new Date().toDateString() === new Date(med.lastTaken).toDateString(),
                        })),
                      )
                      .sort((a, b) => a.reminderTime.localeCompare(b.reminderTime))
                      .map((item, index) => (
                        <div
                          key={`${item.id}-${item.reminderTime}`}
                          className="flex items-center justify-between p-3 glass rounded"
                        >
                          <div className="flex items-center space-x-3">
                            <div
                              className={`w-3 h-3 rounded-full ${item.taken ? "bg-green-500" : "bg-yellow-500"}`}
                            ></div>
                            <div>
                              <div className="font-medium text-sm">{item.reminderTime}</div>
                              <div className="text-xs text-muted-foreground">
                                {item.name} - {item.dose}
                              </div>
                            </div>
                          </div>
                          <Badge variant={item.taken ? "default" : "outline"} className="glass text-xs">
                            {item.taken ? "Taken" : "Pending"}
                          </Badge>
                        </div>
                      ))}
                  </div>
                </CardContent>
              </Card>

              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="text-lg">Medication Insights</CardTitle>
                  <CardDescription>Your adherence patterns and trends</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Overall Adherence</span>
                      <span className="font-medium">87%</span>
                    </div>
                    <Progress value={87} />

                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Doses taken this week:</span>
                        <span>12/14</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Missed doses:</span>
                        <span>2</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Best time compliance:</span>
                        <span>Morning (95%)</span>
                      </div>
                    </div>

                    <div className="mt-4 p-3 glass rounded">
                      <div className="text-sm font-medium mb-1">Reminder</div>
                      <div className="text-xs text-muted-foreground">
                        You've been consistent with morning medications. Consider setting a phone alarm for evening
                        doses to improve adherence.
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="sleep" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Moon className="h-5 w-5 mr-2" />
                Sleep Tracking
              </CardTitle>
              <CardDescription>Track your sleep patterns and quality</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Bedtime</label>
                    <Input
                      type="time"
                      value={bedTime}
                      onChange={(e) => setBedTime(e.target.value)}
                      className="glass mt-2"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium">Wake Time</label>
                    <Input
                      type="time"
                      value={wakeTime}
                      onChange={(e) => setWakeTime(e.target.value)}
                      className="glass mt-2"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium">Sleep Quality (1-10)</label>
                    <div className="grid grid-cols-5 gap-2 mt-2">
                      {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((quality) => (
                        <Button
                          key={quality}
                          variant={quality === sleepQuality ? "default" : "outline"}
                          className={`glass h-10 ${quality === sleepQuality ? "bg-primary text-primary-foreground" : ""}`}
                          size="sm"
                          onClick={() => setSleepQuality(quality)}
                        >
                          {quality}
                        </Button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium">Sleep Notes (Optional)</label>
                    <Textarea
                      value={sleepNotes}
                      onChange={(e) => setSleepNotes(e.target.value)}
                      placeholder="How did you sleep? Any dreams, interruptions, or observations?"
                      className="glass mt-2"
                      rows={3}
                    />
                  </div>

                  <Button onClick={addSleepEntry} className="w-full glass-strong">
                    <Plus className="h-4 w-4 mr-2" />
                    Log Sleep Entry
                  </Button>
                </div>

                <div className="space-y-4">
                  <Card className="glass">
                    <CardHeader>
                      <CardTitle className="text-lg">Sleep Summary</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Estimated Duration:</span>
                          <span className="font-medium">
                            {(() => {
                              const bedTimeDate = new Date(`2000-01-01T${bedTime}:00`)
                              const wakeTimeDate = new Date(`2000-01-01T${wakeTime}:00`)
                              if (wakeTimeDate < bedTimeDate) {
                                wakeTimeDate.setDate(wakeTimeDate.getDate() + 1)
                              }
                              const duration = (wakeTimeDate.getTime() - bedTimeDate.getTime()) / (1000 * 60 * 60)
                              return `${Math.round(duration * 2) / 2}h`
                            })()}
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Sleep Quality:</span>
                          <span className="font-medium">{sleepQuality}/10</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Bedtime:</span>
                          <span className="font-medium">{bedTime}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Wake Time:</span>
                          <span className="font-medium">{wakeTime}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="glass">
                    <CardHeader>
                      <CardTitle className="text-lg">Sleep Tips</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2 text-sm">
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-primary rounded-full"></div>
                          <span>Aim for 7-9 hours of sleep per night</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-secondary rounded-full"></div>
                          <span>Keep a consistent sleep schedule</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-primary rounded-full"></div>
                          <span>Avoid screens 1 hour before bed</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-secondary rounded-full"></div>
                          <span>Create a relaxing bedtime routine</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </CardContent>
          </Card>

          {moodEntries.length > 0 && (
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Recent Sleep History</CardTitle>
                <CardDescription>Your sleep patterns over the last week</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {moodEntries.slice(0, 7).map((entry) => (
                    <div key={entry.id} className="p-4 glass rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-sm font-medium">{entry.date}</div>
                        <div className="flex space-x-2">
                          <Badge variant="outline" className="glass text-xs">
                            {entry.sleep}h
                          </Badge>
                          <Badge
                            variant="outline"
                            className={`glass text-xs ${
                              entry.sleep >= 7 && entry.sleep <= 9
                                ? "border-green-500 text-green-700"
                                : "border-yellow-500 text-yellow-700"
                            }`}
                          >
                            {entry.sleep >= 7 && entry.sleep <= 9 ? "Optimal" : "Suboptimal"}
                          </Badge>
                        </div>
                      </div>
                      {entry.notes && (
                        <div className="text-sm text-muted-foreground mt-2 p-2 glass rounded">{entry.notes}</div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="calendar" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Calendar className="h-5 w-5 mr-2" />
                Calendar View
              </CardTitle>
              <CardDescription>View your wellness data over time</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-medium mb-4">Recent Mood Trends</h3>
                  <div className="space-y-2">
                    {moodEntries.slice(0, 7).map((entry) => (
                      <div key={entry.id} className="flex items-center justify-between p-2 glass rounded">
                        <span className="text-sm">{entry.date}</span>
                        <div className="flex space-x-2">
                          <Badge variant="outline" className="glass text-xs">
                            Mood: {entry.mood}
                          </Badge>
                          <Badge variant="outline" className="glass text-xs">
                            Anxiety: {entry.anxiety}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className="font-medium mb-4">Sleep Pattern</h3>
                  <div className="space-y-2">
                    {moodEntries.slice(0, 7).map((entry) => (
                      <div key={entry.id} className="flex items-center justify-between p-2 glass rounded">
                        <span className="text-sm">{entry.date}</span>
                        <Badge variant="outline" className="glass text-xs">
                          {entry.sleep}h
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <TrendingUp className="h-5 w-5 mr-2" />
                Advanced Analytics
              </CardTitle>
              <CardDescription>Insights and patterns in your wellness data</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
                <div className="text-center p-4 glass rounded-lg">
                  <div className="text-3xl font-bold text-primary mb-2">
                    {moodEntries.length > 0
                      ? (moodEntries.reduce((sum, entry) => sum + entry.mood, 0) / moodEntries.length).toFixed(1)
                      : "0"}
                  </div>
                  <div className="text-sm text-muted-foreground">Average Mood</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {moodEntries.length > 0 &&
                    moodEntries.reduce((sum, entry) => sum + entry.mood, 0) / moodEntries.length > 7
                      ? "Excellent"
                      : moodEntries.length > 0 &&
                          moodEntries.reduce((sum, entry) => sum + entry.mood, 0) / moodEntries.length > 5
                        ? "Good"
                        : "Needs attention"}
                  </div>
                </div>

                <div className="text-center p-4 glass rounded-lg">
                  <div className="text-3xl font-bold text-secondary mb-2">
                    {moodEntries.length > 0
                      ? (moodEntries.reduce((sum, entry) => sum + entry.sleep, 0) / moodEntries.length).toFixed(1)
                      : "0"}
                    h
                  </div>
                  <div className="text-sm text-muted-foreground">Average Sleep</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {moodEntries.length > 0 &&
                    moodEntries.reduce((sum, entry) => sum + entry.sleep, 0) / moodEntries.length >= 7
                      ? "Healthy range"
                      : "Below recommended"}
                  </div>
                </div>

                <div className="text-center p-4 glass rounded-lg">
                  <div className="text-3xl font-bold text-primary mb-2">
                    {moodEntries.length > 0
                      ? (moodEntries.reduce((sum, entry) => sum + entry.anxiety, 0) / moodEntries.length).toFixed(1)
                      : "0"}
                  </div>
                  <div className="text-sm text-muted-foreground">Average Anxiety</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {moodEntries.length > 0 &&
                    moodEntries.reduce((sum, entry) => sum + entry.anxiety, 0) / moodEntries.length < 3
                      ? "Low levels"
                      : moodEntries.length > 0 &&
                          moodEntries.reduce((sum, entry) => sum + entry.anxiety, 0) / moodEntries.length < 6
                        ? "Moderate"
                        : "High - consider support"}
                  </div>
                </div>

                <div className="text-center p-4 glass rounded-lg">
                  <div className="text-3xl font-bold text-secondary mb-2">
                    {routines.filter((r) => r.completed).length}/{routines.length}
                  </div>
                  <div className="text-sm text-muted-foreground">Routines Completed</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {routines.length > 0 && routines.filter((r) => r.completed).length / routines.length > 0.8
                      ? "Excellent consistency"
                      : "Room for improvement"}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                <Card className="glass">
                  <CardHeader>
                    <CardTitle className="text-lg">Mood Trend (Last 7 Days)</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {moodEntries.slice(0, 7).map((entry, index) => {
                        const moodPercentage = (entry.mood / 10) * 100
                        return (
                          <div key={entry.id} className="flex items-center space-x-3">
                            <div className="text-sm font-medium w-20">{entry.date}</div>
                            <div className="flex-1">
                              <div className="flex items-center space-x-2">
                                <Progress value={moodPercentage} className="flex-1" />
                                <span className="text-sm font-medium w-8">{entry.mood}</span>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </CardContent>
                </Card>

                <Card className="glass">
                  <CardHeader>
                    <CardTitle className="text-lg">Sleep Quality Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Optimal Sleep (7-9h)</span>
                        <span className="text-sm font-medium">
                          {moodEntries.filter((e) => e.sleep >= 7 && e.sleep <= 9).length}/{moodEntries.length} days
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Under 7 hours</span>
                        <span className="text-sm font-medium">
                          {moodEntries.filter((e) => e.sleep < 7).length}/{moodEntries.length} days
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Over 9 hours</span>
                        <span className="text-sm font-medium">
                          {moodEntries.filter((e) => e.sleep > 9).length}/{moodEntries.length} days
                        </span>
                      </div>
                      <div className="mt-4 p-3 glass rounded">
                        <div className="text-sm font-medium mb-1">Sleep Consistency Score</div>
                        <Progress
                          value={
                            moodEntries.length > 0
                              ? (moodEntries.filter((e) => e.sleep >= 7 && e.sleep <= 9).length / moodEntries.length) *
                                100
                              : 0
                          }
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card className="glass mb-6">
                <CardHeader>
                  <CardTitle className="text-lg">Mood-Sleep Correlation</CardTitle>
                  <CardDescription>How your sleep affects your mood</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="text-center p-4 glass rounded-lg">
                      <div className="text-2xl font-bold text-green-500 mb-2">
                        {moodEntries.length > 0 ? moodEntries.filter((e) => e.sleep >= 7 && e.mood >= 7).length : 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Good Sleep + Good Mood</div>
                    </div>
                    <div className="text-center p-4 glass rounded-lg">
                      <div className="text-2xl font-bold text-yellow-500 mb-2">
                        {moodEntries.length > 0 ? moodEntries.filter((e) => e.sleep < 7 && e.mood < 6).length : 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Poor Sleep + Low Mood</div>
                    </div>
                    <div className="text-center p-4 glass rounded-lg">
                      <div className="text-2xl font-bold text-blue-500 mb-2">
                        {moodEntries.length > 0
                          ? Math.round(
                              (moodEntries.filter((e) => e.sleep >= 7 && e.mood >= 7).length /
                                Math.max(moodEntries.length, 1)) *
                                100,
                            )
                          : 0}
                        %
                      </div>
                      <div className="text-sm text-muted-foreground">Positive Correlation</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="glass">
                <CardHeader>
                  <CardTitle className="text-lg">Personalized Insights & Recommendations</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {moodEntries.length > 0 && (
                      <>
                        {moodEntries.reduce((sum, entry) => sum + entry.mood, 0) / moodEntries.length > 7 && (
                          <div className="p-4 glass rounded-lg border-l-4 border-green-500">
                            <div className="flex items-center space-x-2 mb-2">
                              <TrendingUp className="h-5 w-5 text-green-500" />
                              <span className="font-medium text-green-700">Excellent Progress!</span>
                            </div>
                            <p className="text-sm text-muted-foreground">
                              Your mood has been consistently high this week. Keep up the great work with your current
                              routines and habits!
                            </p>
                          </div>
                        )}

                        {moodEntries.reduce((sum, entry) => sum + entry.sleep, 0) / moodEntries.length < 7 && (
                          <div className="p-4 glass rounded-lg border-l-4 border-yellow-500">
                            <div className="flex items-center space-x-2 mb-2">
                              <Moon className="h-5 w-5 text-yellow-500" />
                              <span className="font-medium text-yellow-700">Sleep Improvement Needed</span>
                            </div>
                            <p className="text-sm text-muted-foreground">
                              Your average sleep is below the recommended 7-9 hours. Try establishing a consistent
                              bedtime routine and avoiding screens before bed.
                            </p>
                          </div>
                        )}

                        {moodEntries.reduce((sum, entry) => sum + entry.anxiety, 0) / moodEntries.length > 6 && (
                          <div className="p-4 glass rounded-lg border-l-4 border-red-500">
                            <div className="flex items-center space-x-2 mb-2">
                              <Heart className="h-5 w-5 text-red-500" />
                              <span className="font-medium text-red-700">Anxiety Management</span>
                            </div>
                            <p className="text-sm text-muted-foreground">
                              Your anxiety levels have been elevated. Consider practicing mindfulness, deep breathing
                              exercises, or speaking with a healthcare professional.
                            </p>
                          </div>
                        )}

                        <div className="p-4 glass rounded-lg border-l-4 border-blue-500">
                          <div className="flex items-center space-x-2 mb-2">
                            <Activity className="h-5 w-5 text-blue-500" />
                            <span className="font-medium text-blue-700">Weekly Summary</span>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            You've logged {moodEntries.length} mood entries this week. Consistent tracking helps
                            identify patterns and improve your mental wellness journey.
                          </p>
                        </div>
                      </>
                    )}

                    {moodEntries.length === 0 && (
                      <div className="text-center py-8 text-muted-foreground">
                        <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                        <p>No data available yet</p>
                        <p className="text-sm mt-2">Start tracking your mood to see personalized insights</p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="profile" className="space-y-6">
          {console.log("[v0] Profile tab rendered")}
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <User className="h-5 w-5 mr-2" />
                Profile Settings
              </CardTitle>
              <CardDescription>Manage your personal information and preferences</CardDescription>
            </CardHeader>
            <CardContent>
              {isEditingProfile ? (
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="name">Name</Label>
                    <Input
                      id="name"
                      value={tempProfileData.name}
                      onChange={(e) => setTempProfileData({ ...tempProfileData, name: e.target.value })}
                      className="glass-input"
                    />
                  </div>
                  <div>
                    <Label htmlFor="age">Age</Label>
                    <Input
                      id="age"
                      type="number"
                      value={tempProfileData.age}
                      onChange={(e) =>
                        setTempProfileData({
                          ...tempProfileData,
                          age: Number.parseInt(e.target.value, 10) || 0,
                        })
                      }
                      className="glass-input"
                    />
                  </div>
                  <div>
                    <Label htmlFor="gender">Gender</Label>
                    <Select
                      value={tempProfileData.gender}
                      onValueChange={(value) =>
                        setTempProfileData({ ...tempProfileData, gender: value as ProfileData["gender"] })
                      }
                    >
                      <SelectTrigger className="glass-input">
                        <SelectValue placeholder="Select gender" />
                      </SelectTrigger>
                      <SelectContent className="glass-card">
                        <SelectItem value="male">Male</SelectItem>
                        <SelectItem value="female">Female</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                        <SelectItem value="non-binary">Non-Binary</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Known Conditions</Label>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {tempProfileData.conditions.map((condition) => (
                        <Badge key={condition} variant="secondary" className="glass">
                          {condition}
                          <Button
                            variant="ghost"
                            size="icon"
                            className="ml-2 -mr-1 h-4 w-4 glass"
                            onClick={() => removeCondition(condition)}
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        </Badge>
                      ))}
                      <div className="flex items-center space-x-2">
                        <Input
                          type="text"
                          placeholder="Add condition"
                          value={newCondition}
                          onChange={(e) => setNewCondition(e.target.value)}
                          className="glass-input w-32"
                        />
                        <Button variant="outline" size="sm" className="glass bg-transparent" onClick={addCondition}>
                          <Plus className="h-4 w-4 mr-2" />
                          Add
                        </Button>
                      </div>
                    </div>
                  </div>
                  <div>
                    <Label htmlFor="emergencyContact">Emergency Contact</Label>
                    <Input
                      id="emergencyContact"
                      value={tempProfileData.emergencyContact}
                      onChange={(e) =>
                        setTempProfileData({
                          ...tempProfileData,
                          emergencyContact: e.target.value,
                        })
                      }
                      className="glass-input"
                    />
                  </div>
                  <div>
                    <Label htmlFor="email">Email</Label>
                    <Input
                      id="email"
                      type="email"
                      value={tempProfileData.email}
                      onChange={(e) => setTempProfileData({ ...tempProfileData, email: e.target.value })}
                      className="glass-input"
                    />
                  </div>
                  <div className="flex justify-end space-x-2">
                    <Button variant="ghost" className="glass" onClick={cancelProfileEdit}>
                      Cancel
                    </Button>
                    <Button className="glass-strong" onClick={saveProfile}>
                      Update Profile
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div>
                    <div className="text-sm font-medium">Name</div>
                    <div className="text-muted-foreground">{profileData.name || "Not set"}</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium">Age</div>
                    <div className="text-muted-foreground">{profileData.age || "Not set"}</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium">Gender</div>
                    <div className="text-muted-foreground">{profileData.gender || "Not set"}</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium">Known Conditions</div>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {profileData.conditions.length > 0 ? (
                        profileData.conditions.map((condition) => (
                          <Badge key={condition} variant="secondary" className="glass">
                            {condition}
                          </Badge>
                        ))
                      ) : (
                        <div className="text-muted-foreground text-sm">No conditions added</div>
                      )}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm font-medium">Emergency Contact</div>
                    <div className="text-muted-foreground">{profileData.emergencyContact || "Not set"}</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium">Email</div>
                    <div className="text-muted-foreground">{profileData.email || "Not set"}</div>
                  </div>
                  <Button className="w-full glass-strong" onClick={() => setIsEditingProfile(true)}>
                    Edit Profile
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="chat" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <MessageCircle className="h-5 w-5 mr-2" />
                AI Wellness Assistant
              </CardTitle>
              <CardDescription>Chat with your personal wellness assistant for insights and support</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="h-96 overflow-y-auto p-4 glass rounded-lg space-y-4">
                  {chatMessages.length === 0 ? (
                    <div className="text-center text-muted-foreground py-8">
                      <MessageCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Start a conversation with your AI wellness assistant</p>
                      <p className="text-sm mt-2">Ask about mood patterns, sleep tips, or wellness advice</p>
                    </div>
                  ) : (
                    chatMessages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`max-w-[80%] p-3 rounded-lg ${
                            message.role === "user" ? "bg-primary text-primary-foreground ml-4" : "glass mr-4"
                          }`}
                        >
                          <div className="text-sm">{message.content}</div>
                          <div className="text-xs opacity-70 mt-1">
                            {new Date(message.timestamp).toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>

                <div className="flex space-x-2">
                  <Input
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Ask about your mood, sleep, or wellness..."
                    className="glass-input"
                    onKeyPress={(e) => e.key === "Enter" && sendChatMessage()}
                  />
                  <Button onClick={sendChatMessage} className="glass-strong" disabled={!chatInput.trim()}>
                    <MessageCircle className="h-4 w-4" />
                  </Button>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="glass text-xs bg-transparent"
                    onClick={() => {
                      setChatInput("How has my mood been trending lately?")
                      sendChatMessage()
                    }}
                  >
                    Mood trends
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="glass text-xs bg-transparent"
                    onClick={() => {
                      setChatInput("Give me sleep improvement tips")
                      sendChatMessage()
                    }}
                  >
                    Sleep tips
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="glass text-xs bg-transparent"
                    onClick={() => {
                      setChatInput("Help me manage anxiety")
                      sendChatMessage()
                    }}
                  >
                    Anxiety help
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="glass text-xs bg-transparent"
                    onClick={() => {
                      setChatInput("What wellness activities should I try?")
                      sendChatMessage()
                    }}
                  >
                    Wellness tips
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="integrations" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Plug className="h-5 w-5 mr-2" />
                API Integrations
              </CardTitle>
              <CardDescription>Connect MindTrack with your other apps and services</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* API Keys Section */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">API Keys</h3>
                    <Button
                      onClick={() => {
                        const newKey: ApiKey = {
                          id: Date.now().toString(),
                          name: `API Key ${apiKeys.length + 1}`,
                          key: `mt_${Math.random().toString(36).substring(2, 15)}${Math.random().toString(36).substring(2, 15)}`,
                          createdAt: new Date().toISOString(),
                          permissions: ["read", "write"],
                        }
                        setApiKeys([...apiKeys, newKey])
                      }}
                      className="glass-strong"
                    >
                      <Plus className="h-4 w-4 mr-2" />
                      Generate New Key
                    </Button>
                  </div>

                  {apiKeys.length === 0 ? (
                    <div className="text-center py-8 glass rounded-lg">
                      <Plug className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p className="text-muted-foreground">No API keys yet</p>
                      <p className="text-sm text-muted-foreground mt-2">
                        Generate an API key to start integrating with other apps
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {apiKeys.map((key) => (
                        <div key={key.id} className="glass p-4 rounded-lg space-y-3">
                          <div className="flex items-center justify-between">
                            <div className="space-y-1">
                              <Input
                                value={key.name}
                                onChange={(e) => {
                                  setApiKeys(apiKeys.map((k) => (k.id === key.id ? { ...k, name: e.target.value } : k)))
                                }}
                                className="font-medium bg-transparent border-none p-0 h-auto focus-visible:ring-0"
                              />
                              <p className="text-xs text-muted-foreground">
                                Created {new Date(key.createdAt).toLocaleDateString()}
                              </p>
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                setApiKeys(apiKeys.filter((k) => k.id !== key.id))
                              }}
                            >
                              <Trash2 className="h-4 w-4 text-destructive" />
                            </Button>
                          </div>

                          <div className="flex items-center space-x-2">
                            <Input value={key.key} readOnly className="font-mono text-xs glass-input" />
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                navigator.clipboard.writeText(key.key)
                              }}
                              className="glass"
                            >
                              <Copy className="h-4 w-4" />
                            </Button>
                          </div>

                          <div className="flex flex-wrap gap-2">
                            {key.permissions.map((permission) => (
                              <Badge key={permission} variant="secondary" className="text-xs">
                                {permission}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* API Documentation Preview */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Available Endpoints</h3>
                  <div className="space-y-2">
                    <div className="glass p-3 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <Badge variant="outline" className="mr-2">
                            GET
                          </Badge>
                          <code className="text-sm">/api/v1/mood</code>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => setSelectedTab("integrations-docs")}>
                          View Docs
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">Retrieve mood tracking data</p>
                    </div>

                    <div className="glass p-3 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <Badge variant="outline" className="mr-2">
                            GET
                          </Badge>
                          <code className="text-sm">/api/v1/sleep</code>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => setSelectedTab("integrations-docs")}>
                          View Docs
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">Retrieve sleep tracking data</p>
                    </div>

                    <div className="glass p-3 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <Badge variant="outline" className="mr-2">
                            GET
                          </Badge>
                          <code className="text-sm">/api/v1/medications</code>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => setSelectedTab("integrations-docs")}>
                          View Docs
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">Retrieve medication data</p>
                    </div>

                    <div className="glass p-3 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <Badge variant="outline" className="mr-2">
                            GET
                          </Badge>
                          <code className="text-sm">/api/v1/analytics</code>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => setSelectedTab("integrations-docs")}>
                          View Docs
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">Retrieve analytics and insights</p>
                    </div>
                  </div>
                </div>

                {/* Webhook Configuration */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Webhooks</h3>
                  <div className="glass p-4 rounded-lg space-y-3">
                    <p className="text-sm text-muted-foreground">
                      Configure webhooks to receive real-time updates when your data changes
                    </p>
                    <div className="space-y-2">
                      <Label>Webhook URL</Label>
                      <Input placeholder="https://your-app.com/webhook" className="glass-input" />
                    </div>
                    <div className="space-y-2">
                      <Label>Events</Label>
                      <div className="flex flex-wrap gap-2">
                        <Badge variant="outline" className="cursor-pointer">
                          mood.created
                        </Badge>
                        <Badge variant="outline" className="cursor-pointer">
                          sleep.created
                        </Badge>
                        <Badge variant="outline" className="cursor-pointer">
                          medication.taken
                        </Badge>
                      </div>
                    </div>
                    <Button className="glass-strong w-full">Save Webhook</Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="integrations-docs" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center">
                    <FileText className="h-5 w-5 mr-2" />
                    API Documentation
                  </CardTitle>
                  <CardDescription>Complete guide to integrating with MindTrack</CardDescription>
                </div>
                <Button variant="outline" onClick={() => setSelectedTab("integrations")} className="glass">
                  Back to Integrations
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* Authentication */}
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold">Authentication</h3>
                  <div className="glass p-4 rounded-lg space-y-2">
                    <p className="text-sm">All API requests require authentication using an API key in the header:</p>
                    <pre className="bg-black/20 p-3 rounded text-xs overflow-x-auto">
                      <code>{`curl -H "x-api-key: YOUR_API_KEY" \\
  https://your-domain.com/api/v1/mood`}</code>
                    </pre>
                  </div>
                </div>

                {/* Mood Endpoint */}
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold">Mood Tracking</h3>
                  <div className="glass p-4 rounded-lg space-y-3">
                    <div>
                      <Badge variant="outline" className="mr-2">
                        GET
                      </Badge>
                      <code className="text-sm">/api/v1/mood</code>
                    </div>
                    <p className="text-sm text-muted-foreground">Retrieve mood tracking entries</p>
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Query Parameters:</p>
                      <ul className="text-xs space-y-1 ml-4">
                        <li>
                          <code>startDate</code> - Filter by start date (YYYY-MM-DD)
                        </li>
                        <li>
                          <code>endDate</code> - Filter by end date (YYYY-MM-DD)
                        </li>
                        <li>
                          <code>limit</code> - Number of results (default: 100)
                        </li>
                      </ul>
                    </div>
                    <pre className="bg-black/20 p-3 rounded text-xs overflow-x-auto">
                      <code>{`{
  "data": [
    {
      "id": "1",
      "date": "2024-01-15",
      "mood": 8,
      "anxiety": 3,
      "energy": 7,
      "notes": "Feeling great today",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "meta": {
    "total": 1,
    "limit": 100
  }
}`}</code>
                    </pre>
                  </div>

                  <div className="glass p-4 rounded-lg space-y-3">
                    <div>
                      <Badge variant="outline" className="mr-2">
                        POST
                      </Badge>
                      <code className="text-sm">/api/v1/mood</code>
                    </div>
                    <p className="text-sm text-muted-foreground">Create a new mood entry</p>
                    <pre className="bg-black/20 p-3 rounded text-xs overflow-x-auto">
                      <code>{`{
  "mood": 8,
  "anxiety": 3,
  "energy": 7,
  "notes": "Feeling great today"
}`}</code>
                    </pre>
                  </div>
                </div>

                {/* Sleep Endpoint */}
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold">Sleep Tracking</h3>
                  <div className="glass p-4 rounded-lg space-y-3">
                    <div>
                      <Badge variant="outline" className="mr-2">
                        GET
                      </Badge>
                      <code className="text-sm">/api/v1/sleep</code>
                    </div>
                    <p className="text-sm text-muted-foreground">Retrieve sleep tracking entries</p>
                    <pre className="bg-black/20 p-3 rounded text-xs overflow-x-auto">
                      <code>{`{
  "data": [
    {
      "id": "1",
      "date": "2024-01-15",
      "bedtime": "23:00",
      "wakeTime": "07:30",
      "quality": 8,
      "duration": 8.5,
      "notes": "Slept well"
    }
  ]
}`}</code>
                    </pre>
                  </div>
                </div>

                {/* Analytics Endpoint */}
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold">Analytics</h3>
                  <div className="glass p-4 rounded-lg space-y-3">
                    <div>
                      <Badge variant="outline" className="mr-2">
                        GET
                      </Badge>
                      <code className="text-sm">/api/v1/analytics</code>
                    </div>
                    <p className="text-sm text-muted-foreground">Retrieve analytics and insights</p>
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Query Parameters:</p>
                      <ul className="text-xs space-y-1 ml-4">
                        <li>
                          <code>period</code> - Time period (week, month, year)
                        </li>
                      </ul>
                    </div>
                    <pre className="bg-black/20 p-3 rounded text-xs overflow-x-auto">
                      <code>{`{
  "period": "week",
  "summary": {
    "averageMood": 7.5,
    "averageAnxiety": 3.2,
    "averageSleep": 7.5
  },
  "insights": [
    "Your mood improves with better sleep"
  ]
}`}</code>
                    </pre>
                  </div>
                </div>

                {/* Rate Limits */}
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold">Rate Limits</h3>
                  <div className="glass p-4 rounded-lg">
                    <p className="text-sm">API requests are limited to:</p>
                    <ul className="text-sm space-y-1 ml-4 mt-2">
                      <li>1000 requests per hour</li>
                      <li>10,000 requests per day</li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Emergency Alerts Tab */}
        <TabsContent value="emergency" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <AlertCircle className="h-5 w-5 mr-2 text-red-500" />
                Emergency Alerts
              </CardTitle>
              <CardDescription>
                Track and manage emergency situations. In a production app, this would connect to emergency services.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4">
                  <p className="text-sm text-yellow-600 dark:text-yellow-400">
                    ⚠️ This feature is for demonstration purposes. In a real app, this would connect to emergency services.
                  </p>
                </div>

                <Dialog>
                  <DialogTrigger asChild>
                    <Button className="glass-strong w-full">
                      <Plus className="h-4 w-4 mr-2" />
                      Create Emergency Alert
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="glass-card">
                    <DialogHeader>
                      <DialogTitle>Create Emergency Alert</DialogTitle>
                      <DialogDescription>Report an emergency situation</DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4">
                      <div>
                        <Label>Alert Type</Label>
                        <Select
                          defaultValue="high_anxiety"
                          onValueChange={(value) => {
                            // Handle alert type selection
                          }}
                        >
                          <SelectTrigger className="glass">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="high_anxiety">High Anxiety</SelectItem>
                            <SelectItem value="suicidal_thoughts">Suicidal Thoughts</SelectItem>
                            <SelectItem value="severe_migraine">Severe Migraine</SelectItem>
                            <SelectItem value="medication_overdose">Medication Overdose</SelectItem>
                            <SelectItem value="panic_attack">Panic Attack</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label>Severity</Label>
                        <Select defaultValue="medium">
                          <SelectTrigger className="glass">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="low">Low</SelectItem>
                            <SelectItem value="medium">Medium</SelectItem>
                            <SelectItem value="high">High</SelectItem>
                            <SelectItem value="critical">Critical</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label>What triggered this alert?</Label>
                        <Textarea className="glass" placeholder="Describe what triggered this emergency..." />
                      </div>
                      <div>
                        <Label>Additional Notes</Label>
                        <Textarea className="glass" placeholder="Any additional information..." />
                      </div>
                      <Button className="w-full glass-strong bg-red-500 hover:bg-red-600">
                        🚨 Create Alert
                      </Button>
                    </div>
                  </DialogContent>
                </Dialog>

                <div className="space-y-3">
                  <h3 className="text-lg font-semibold">Recent Alerts</h3>
                  {emergencyAlerts.length === 0 ? (
                    <div className="text-center py-8 glass rounded-lg">
                      <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p className="text-muted-foreground">No emergency alerts yet</p>
                    </div>
                  ) : (
                    emergencyAlerts.map((alert) => (
                      <Card key={alert.id} className="glass-card">
                        <CardHeader>
                          <div className="flex items-center justify-between">
                            <CardTitle className="text-base">
                              {alert.alertType.replace("_", " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                            </CardTitle>
                            <Badge
                              variant={
                                alert.severity === "critical"
                                  ? "destructive"
                                  : alert.severity === "high"
                                    ? "destructive"
                                    : alert.severity === "medium"
                                      ? "default"
                                      : "secondary"
                              }
                            >
                              {alert.severity.toUpperCase()}
                            </Badge>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            <p className="text-sm">
                              <span className="font-medium">Triggered by:</span> {alert.triggeredBy}
                            </p>
                            <p className="text-sm">
                              <span className="font-medium">Status:</span>{" "}
                              <Badge variant={alert.status === "active" ? "destructive" : "secondary"}>
                                {alert.status}
                              </Badge>
                            </p>
                            <p className="text-sm">
                              <span className="font-medium">Created:</span>{" "}
                              {new Date(alert.createdAt).toLocaleString()}
                            </p>
                            {alert.notes && <p className="text-sm text-muted-foreground">{alert.notes}</p>}
                            {alert.status === "active" && (
                              <div className="flex gap-2 mt-4">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="glass"
                                  onClick={() => {
                                    setEmergencyAlerts(
                                      emergencyAlerts.map((a) =>
                                        a.id === alert.id ? { ...a, status: "acknowledged" } : a,
                                      ),
                                    )
                                  }}
                                >
                                  ✅ Acknowledge
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="glass"
                                  onClick={() => {
                                    setEmergencyAlerts(
                                      emergencyAlerts.map((a) => (a.id === alert.id ? { ...a, status: "resolved" } : a)),
                                    )
                                  }}
                                >
                                  🔒 Resolve
                                </Button>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Weather Correlation Tab */}
        <TabsContent value="weather" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Cloud className="h-5 w-5 mr-2" />
                Weather Correlation
              </CardTitle>
              <CardDescription>Analyze how weather conditions affect your symptoms and mood</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {weatherData.slice(-7).map((weather) => (
                    <Card key={weather.id} className="glass-card">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">{new Date(weather.date).toLocaleDateString()}</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Condition:</span>
                            <Badge variant="outline" className="glass">
                              {weather.condition}
                            </Badge>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">High:</span>
                            <span className="text-sm font-medium">{weather.temperatureHigh.toFixed(1)}°C</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Low:</span>
                            <span className="text-sm font-medium">{weather.temperatureLow.toFixed(1)}°C</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">Humidity:</span>
                            <span className="text-sm font-medium">{weather.humidity.toFixed(0)}%</span>
                          </div>
                          {weather.precipitation > 0 && (
                            <div className="flex items-center justify-between">
                              <span className="text-sm text-muted-foreground">Precipitation:</span>
                              <span className="text-sm font-medium">{weather.precipitation.toFixed(1)}mm</span>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>

                <div className="glass p-4 rounded-lg">
                  <h3 className="text-lg font-semibold mb-4">Correlation Analysis</h3>
                  <p className="text-sm text-muted-foreground">
                    Weather data can help identify patterns in your symptoms. For example, some people experience
                    migraines during weather changes, or mood changes with temperature shifts.
                  </p>
                  <p className="text-sm text-muted-foreground mt-2">
                    Track your symptoms alongside weather data to discover correlations that may affect your mental
                    health.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Data Export Tab */}
        <TabsContent value="export" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Download className="h-5 w-5 mr-2" />
                Data Export
              </CardTitle>
              <CardDescription>Export your data in various formats for backup or analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card className="glass-card">
                    <CardHeader>
                      <CardTitle className="text-base">Export as JSON</CardTitle>
                      <CardDescription>Download all your data as a JSON file</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Button
                        className="w-full glass-strong"
                        onClick={() => {
                          const data = {
                            moodEntries,
                            bodySymptoms,
                            routines,
                            medications,
                            emergencyAlerts,
                            weatherData,
                            profile: profileData,
                            exportedAt: new Date().toISOString(),
                          }
                          const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
                          const url = URL.createObjectURL(blob)
                          const a = document.createElement("a")
                          a.href = url
                          a.download = `mindtrack-export-${new Date().toISOString().split("T")[0]}.json`
                          a.click()
                          URL.revokeObjectURL(url)
                        }}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Download JSON
                      </Button>
                    </CardContent>
                  </Card>

                  <Card className="glass-card">
                    <CardHeader>
                      <CardTitle className="text-base">Export as CSV</CardTitle>
                      <CardDescription>Download mood entries as a CSV file</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Button
                        className="w-full glass-strong"
                        onClick={() => {
                          const csv = [
                            ["Date", "Mood", "Anxiety", "Sleep", "Energy", "Notes"],
                            ...moodEntries.map((entry) => [
                              entry.date,
                              entry.mood.toString(),
                              entry.anxiety.toString(),
                              entry.sleep.toString(),
                              entry.energy.toString(),
                              entry.notes,
                            ]),
                          ]
                            .map((row) => row.map((cell) => `"${cell}"`).join(","))
                            .join("\n")

                          const blob = new Blob([csv], { type: "text/csv" })
                          const url = URL.createObjectURL(blob)
                          const a = document.createElement("a")
                          a.href = url
                          a.download = `mindtrack-mood-entries-${new Date().toISOString().split("T")[0]}.csv`
                          a.click()
                          URL.revokeObjectURL(url)
                        }}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Download CSV
                      </Button>
                    </CardContent>
                  </Card>
                </div>

                <div className="glass p-4 rounded-lg">
                  <h3 className="text-lg font-semibold mb-2">Export Options</h3>
                  <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
                    <li>JSON format includes all data types (mood, sleep, medications, routines, etc.)</li>
                    <li>CSV format is optimized for spreadsheet analysis</li>
                    <li>All exports include timestamps and metadata</li>
                    <li>Your data is processed locally - nothing is sent to external servers</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
