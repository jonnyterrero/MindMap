"""
MindMap – single-file FastAPI app
Tracks sleep, routines, focus/productivity, meds (encrypted optional), migraines,
simple trigger rule, streaks scoring, calendar heatmap, and CSV import for wearables.

Run:
  pip install fastapi uvicorn SQLAlchemy pydantic pandas python-multipart cryptography matplotlib
  uvicorn mindmap:app --reload
"""

from __future__ import annotations
import io
import os
import json
import requests
import schedule
import time
import threading
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field, ConfigDict
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import (
    create_engine, select, Column, Integer, Float, String, Boolean, Date, DateTime,
    JSON, LargeBinary, func, Index
)
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session

# ---------------------------
# Config
# ---------------------------
LOCAL_FIRST: bool = (os.getenv("LOCAL_FIRST", "true").lower() == "true")
ENCRYPTION_KEY: Optional[str] = os.getenv("ENCRYPTION_KEY", None)
TRIGGER_SLEEP_HOURS_LT: float = float(os.getenv("TRIGGER_SLEEP_HOURS_LT", "6.0"))
TRIGGER_ANXIETY_GE: int = int(os.getenv("TRIGGER_ANXIETY_GE", "6"))

# ---------------------------
# DB setup (SQLite, local-first)
# ---------------------------
SQLALCHEMY_DATABASE_URL = "sqlite:///./mindmap.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)

    # Basic Information
    name = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)  # 'male', 'female', 'non-binary', 'other'

    # Medical Information
    known_conditions = Column(JSON, nullable=True)  # list[str] - conditions like "migraine", "anxiety", "depression"
    symptoms = Column(JSON, nullable=True)  # list[str] - recurring symptoms
    allergies = Column(JSON, nullable=True)  # list[str] - allergies

    # Current Medications/Remedies
    current_medications = Column(JSON,
                                 nullable=True)  # list[dict] - {"name": "med_name", "dose": "10mg", "frequency": "daily"}
    supplements = Column(JSON,
                         nullable=True)  # list[dict] - {"name": "vitamin_d", "dose": "1000iu", "frequency": "daily"}

    # Preferences
    preferred_interventions = Column(JSON, nullable=True)  # list[str] - preferred intervention types
    emergency_contact = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Routine(Base):
    __tablename__ = "routines"

    id = Column(Integer, primary_key=True, index=True)
    entry_id = Column(Integer, nullable=False)  # Link to Entry

    # Routine Information
    routine_type = Column(String, nullable=False)  # 'morning', 'work', 'school', 'exercise', 'afternoon', 'evening'
    start_time = Column(String, nullable=True)  # 'HH:MM'
    end_time = Column(String, nullable=True)  # 'HH:MM'
    duration_minutes = Column(Integer, nullable=True)

    # Activities and Details
    activities = Column(JSON, nullable=True)  # list[str] - activities performed
    notes = Column(String, nullable=True)  # additional notes

    # Effectiveness Tracking
    symptoms_improved = Column(JSON, nullable=True)  # list[str] - symptoms that improved
    emotions_improved = Column(JSON, nullable=True)  # list[str] - emotions that improved
    sensations_improved = Column(JSON, nullable=True)  # list[str] - body sensations that improved

    # Effectiveness Rating (0-10 scale)
    overall_effectiveness = Column(Integer, nullable=True)  # 0-10 scale
    energy_level_after = Column(Integer, nullable=True)  # 0-10 scale
    mood_improvement = Column(Integer, nullable=True)  # 0-10 scale
    symptom_relief = Column(Integer, nullable=True)  # 0-10 scale

    # Completion Status
    completed = Column(Boolean, default=True)
    completion_percentage = Column(Integer, nullable=True)  # 0-100

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class MedicationReminder(Base):
    __tablename__ = "medication_reminders"

    id = Column(Integer, primary_key=True, index=True)

    # Medication Information
    medication_name = Column(String, nullable=False)
    dose = Column(String, nullable=True)
    frequency = Column(String, nullable=False)  # 'daily', 'twice_daily', 'three_times_daily', 'weekly', 'custom'
    custom_schedule = Column(JSON, nullable=True)  # For custom schedules

    # Reminder Settings
    reminder_times = Column(JSON, nullable=False)  # list[str] - times like ["08:00", "20:00"]
    enabled = Column(Boolean, default=True)
    notification_method = Column(String, default="app")  # 'app', 'email', 'sms'

    # Tracking
    last_taken = Column(DateTime(timezone=True), nullable=True)
    next_reminder = Column(DateTime(timezone=True), nullable=True)
    missed_doses = Column(Integer, default=0)

    # Notes
    notes = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class WeatherData(Base):
    __tablename__ = "weather_data"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True, nullable=False)

    # Weather Information
    temperature_high = Column(Float, nullable=True)
    temperature_low = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    pressure = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    precipitation = Column(Float, nullable=True)
    weather_condition = Column(String, nullable=True)  # 'sunny', 'rainy', 'cloudy', etc.

    # Location
    location = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # Correlation Data
    symptom_correlation = Column(JSON, nullable=True)  # Correlation with user symptoms

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class EmergencyAlert(Base):
    __tablename__ = "emergency_alerts"

    id = Column(Integer, primary_key=True, index=True)

    # Alert Information
    alert_type = Column(String,
                        nullable=False)  # 'high_anxiety', 'suicidal_thoughts', 'severe_migraine', 'medication_overdose'
    severity = Column(String, nullable=False)  # 'low', 'medium', 'high', 'critical'
    triggered_by = Column(String, nullable=False)  # What triggered the alert

    # Alert Data
    trigger_data = Column(JSON, nullable=True)  # Data that triggered the alert
    user_notes = Column(String, nullable=True)

    # Status
    status = Column(String, default="active")  # 'active', 'acknowledged', 'resolved'
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # Response
    response_action = Column(String, nullable=True)  # What action was taken
    contacted_emergency_services = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=False, unique=True)

    # Session Information
    user_context = Column(JSON, nullable=True)  # User's current context (mood, symptoms, etc.)
    conversation_history = Column(JSON, nullable=True)  # Chat history

    # Session Status
    active = Column(Boolean, default=True)
    last_activity = Column(DateTime(timezone=True), server_default=func.now())

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------
# Field-level encryption (optional)
# ---------------------------
_fernet: Optional[Fernet] = Fernet(ENCRYPTION_KEY) if ENCRYPTION_KEY else None


def enc(text: Optional[str]) -> Optional[bytes]:
    if text is None:
        return None
    if _fernet is None:
        return text.encode("utf-8")
    return _fernet.encrypt(text.encode("utf-8"))


def dec(blob: Optional[bytes]) -> Optional[str]:
    if blob is None:
        return None
    if _fernet is None:
        try:
            return blob.decode("utf-8")
        except Exception:
            return None
    try:
        return _fernet.decrypt(blob).decode("utf-8")
    except InvalidToken:
        return None


# ---------------------------
# ORM model
# ---------------------------
class Entry(Base):
    __tablename__ = "entries"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True, nullable=False)

    # Sleep
    sleep_hours = Column(Float, nullable=True)  # 0..24
    sleep_quality = Column(Integer, nullable=True)  # 1..5
    bed_time = Column(String, nullable=True)  # 'HH:MM'
    wake_time = Column(String, nullable=True)  # 'HH:MM'
    hrv = Column(Integer, nullable=True)  # ms or index

    # Mental-health scales
    mood_valence = Column(Integer, nullable=True)  # -3..+3
    anxiety_level = Column(Integer, nullable=True)  # 0..10
    depression_level = Column(Integer, nullable=True)  # 0..10
    mania_level = Column(Integer, nullable=True)  # 0..10

    # ADHD / Focus / Productivity
    adhd_focus = Column(Integer, nullable=True)  # 0..10
    productivity_score = Column(Integer, nullable=True)  # 0..100

    # Routines
    routines_followed = Column(JSON, nullable=True)  # list[str]

    # Migraine
    migraine = Column(Boolean, nullable=False, default=False)
    migraine_intensity = Column(Integer, nullable=True)  # 0..10
    migraine_aura = Column(Boolean, nullable=True)

    # Body map sensations (JSON: {"head": 5, "chest": 3, "stomach": 7, ...})
    body_sensations = Column(JSON, nullable=True)  # dict[str, int] - body part -> intensity 0-10

    # Sensitive at-rest (optional encrypted)
    meds_cipher = Column(LargeBinary, nullable=True)  # bytes
    notes_cipher = Column(LargeBinary, nullable=True)  # bytes

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


Index("idx_entries_date", Entry.date)


# ---------------------------
# Schemas
# ---------------------------
class Med(BaseModel):
    name: str
    dose_mg: Optional[float] = None
    at_time: Optional[str] = None  # 'HH:MM'


class EntryBase(BaseModel):
    date: date
    sleep_hours: Optional[float] = Field(None, ge=0, le=24)
    sleep_quality: Optional[int] = Field(None, ge=1, le=5)
    bed_time: Optional[str] = None
    wake_time: Optional[str] = None
    hrv: Optional[int] = Field(None, ge=0, le=400)

    mood_valence: Optional[int] = Field(None, ge=-3, le=3)
    anxiety_level: Optional[int] = Field(None, ge=0, le=10)
    depression_level: Optional[int] = Field(None, ge=0, le=10)
    mania_level: Optional[int] = Field(None, ge=0, le=10)

    adhd_focus: Optional[int] = Field(None, ge=0, le=10)
    productivity_score: Optional[int] = Field(None, ge=0, le=100)

    routines_followed: Optional[List[str]] = None

    migraine: bool = False
    migraine_intensity: Optional[int] = Field(None, ge=0, le=10)
    migraine_aura: Optional[bool] = None

    body_sensations: Optional[Dict[str, int]] = Field(None, description="Body part -> intensity mapping")

    meds: Optional[List[Med]] = None
    notes: Optional[str] = None


class EntryCreate(EntryBase):
    pass


class EntryUpdate(EntryBase):
    pass


class EntryOut(EntryBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)


class MedicationItem(BaseModel):
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    notes: Optional[str] = None


class UserProfileBase(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = None

    known_conditions: Optional[List[str]] = None
    symptoms: Optional[List[str]] = None
    allergies: Optional[List[str]] = None

    current_medications: Optional[List[MedicationItem]] = None
    supplements: Optional[List[MedicationItem]] = None

    preferred_interventions: Optional[List[str]] = None
    emergency_contact: Optional[str] = None


class UserProfileCreate(UserProfileBase):
    pass


class UserProfileUpdate(UserProfileBase):
    pass


class UserProfileOut(UserProfileBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)


class RoutineBase(BaseModel):
    entry_id: int
    routine_type: str  # 'morning', 'work', 'school', 'exercise', 'afternoon', 'evening'
    start_time: Optional[str] = None  # 'HH:MM'
    end_time: Optional[str] = None  # 'HH:MM'
    duration_minutes: Optional[int] = Field(None, ge=0, le=1440)  # max 24 hours

    activities: Optional[List[str]] = None
    notes: Optional[str] = None

    symptoms_improved: Optional[List[str]] = None
    emotions_improved: Optional[List[str]] = None
    sensations_improved: Optional[List[str]] = None

    overall_effectiveness: Optional[int] = Field(None, ge=0, le=10)
    energy_level_after: Optional[int] = Field(None, ge=0, le=10)
    mood_improvement: Optional[int] = Field(None, ge=0, le=10)
    symptom_relief: Optional[int] = Field(None, ge=0, le=10)

    completed: bool = True
    completion_percentage: Optional[int] = Field(None, ge=0, le=100)


class RoutineCreate(RoutineBase):
    pass


class RoutineUpdate(RoutineBase):
    pass


class RoutineOut(RoutineBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)


# Medication Reminder Schemas
class MedicationReminderBase(BaseModel):
    medication_name: str
    dose: Optional[str] = None
    frequency: str  # 'daily', 'twice_daily', 'three_times_daily', 'weekly', 'custom'
    custom_schedule: Optional[Dict[str, Any]] = None
    reminder_times: List[str]  # ["08:00", "20:00"]
    enabled: bool = True
    notification_method: str = "app"  # 'app', 'email', 'sms'
    notes: Optional[str] = None


class MedicationReminderCreate(MedicationReminderBase):
    pass


class MedicationReminderUpdate(MedicationReminderBase):
    pass


class MedicationReminderOut(MedicationReminderBase):
    id: int
    last_taken: Optional[datetime] = None
    next_reminder: Optional[datetime] = None
    missed_doses: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)


# Weather Data Schemas
class WeatherDataBase(BaseModel):
    date: date
    temperature_high: Optional[float] = None
    temperature_low: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    precipitation: Optional[float] = None
    weather_condition: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    symptom_correlation: Optional[Dict[str, Any]] = None


class WeatherDataCreate(WeatherDataBase):
    pass


class WeatherDataOut(WeatherDataBase):
    id: int
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)


# Emergency Alert Schemas
class EmergencyAlertBase(BaseModel):
    alert_type: str  # 'high_anxiety', 'suicidal_thoughts', 'severe_migraine', 'medication_overdose'
    severity: str  # 'low', 'medium', 'high', 'critical'
    triggered_by: str
    trigger_data: Optional[Dict[str, Any]] = None
    user_notes: Optional[str] = None
    status: str = "active"  # 'active', 'acknowledged', 'resolved'
    response_action: Optional[str] = None
    contacted_emergency_services: bool = False


class EmergencyAlertCreate(EmergencyAlertBase):
    pass


class EmergencyAlertUpdate(EmergencyAlertBase):
    pass


class EmergencyAlertOut(EmergencyAlertBase):
    id: int
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)


# Chat Session Schemas
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime


class ChatSessionBase(BaseModel):
    user_context: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[ChatMessage]] = None
    active: bool = True


class ChatSessionCreate(ChatSessionBase):
    pass


class ChatSessionOut(ChatSessionBase):
    id: int
    session_id: str
    last_activity: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)


# ---------------------------
# CRUD helpers
# ---------------------------
def _serialize_meds(meds: Optional[List[Med]]) -> Optional[bytes]:
    if meds is None:
        return None
    return enc(json.dumps([m.model_dump() for m in meds]))


def _deserialize_meds(blob: Optional[bytes]) -> Optional[List[Dict]]:
    if blob is None:
        return None
    s = dec(blob)
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _serialize_notes(notes: Optional[str]) -> Optional[bytes]:
    if notes is None:
        return None
    return enc(notes)


def _deserialize_notes(blob: Optional[bytes]) -> Optional[str]:
    if blob is None:
        return None
    return dec(blob)


def entry_to_dict(e: Entry) -> Dict:
    return {
        "id": e.id,
        "date": e.date.isoformat(),
        "sleep_hours": e.sleep_hours,
        "sleep_quality": e.sleep_quality,
        "bed_time": e.bed_time,
        "wake_time": e.wake_time,
        "hrv": e.hrv,
        "mood_valence": e.mood_valence,
        "anxiety_level": e.anxiety_level,
        "depression_level": e.depression_level,
        "mania_level": e.mania_level,
        "adhd_focus": e.adhd_focus,
        "productivity_score": e.productivity_score,
        "routines_followed": e.routines_followed or [],
        "migraine": bool(e.migraine),
        "migraine_intensity": e.migraine_intensity,
        "migraine_aura": e.migraine_aura,
        "body_sensations": e.body_sensations or {},
        "meds": _deserialize_meds(e.meds_cipher),
        "notes": _deserialize_notes(e.notes_cipher),
        "created_at": e.created_at.isoformat() if e.created_at else None,
        "updated_at": e.updated_at.isoformat() if e.updated_at else None,
    }


def create_entry(db: Session, payload: EntryCreate) -> Entry:
    obj = Entry(
        date=payload.date,
        sleep_hours=payload.sleep_hours,
        sleep_quality=payload.sleep_quality,
        bed_time=payload.bed_time,
        wake_time=payload.wake_time,
        hrv=payload.hrv,
        mood_valence=payload.mood_valence,
        anxiety_level=payload.anxiety_level,
        depression_level=payload.depression_level,
        mania_level=payload.mania_level,
        adhd_focus=payload.adhd_focus,
        productivity_score=payload.productivity_score,
        routines_followed=payload.routines_followed,
        migraine=payload.migraine,
        migraine_intensity=payload.migraine_intensity,
        migraine_aura=payload.migraine_aura,
        body_sensations=payload.body_sensations,
        meds_cipher=_serialize_meds(payload.meds),
        notes_cipher=_serialize_notes(payload.notes),
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_entries(
        db: Session,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 5000,
        offset: int = 0,
) -> List[Entry]:
    stmt = select(Entry)
    if start is not None:
        stmt = stmt.where(Entry.date >= start)
    if end is not None:
        stmt = stmt.where(Entry.date <= end)
    stmt = stmt.order_by(Entry.date.asc()).limit(limit).offset(offset)
    return list(db.scalars(stmt).all())


def get_entry(db: Session, entry_id: int) -> Optional[Entry]:
    return db.get(Entry, entry_id)


def update_entry(db: Session, entry_id: int, payload: EntryUpdate) -> Optional[Entry]:
    obj = db.get(Entry, entry_id)
    if not obj:
        return None
    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        if k == "meds":
            obj.meds_cipher = _serialize_meds(v)
        elif k == "notes":
            obj.notes_cipher = _serialize_notes(v)
        else:
            setattr(obj, k, v)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def delete_entry(db: Session, entry_id: int) -> bool:
    obj = db.get(Entry, entry_id)
    if not obj:
        return False
    db.delete(obj)
    db.commit()
    return True


# User Profile CRUD helpers
def get_user_profile(db: Session) -> Optional[UserProfile]:
    """Get the first (and only) user profile."""
    return db.query(UserProfile).first()


def create_user_profile(db: Session, payload: UserProfileCreate) -> UserProfile:
    """Create a new user profile."""
    obj = UserProfile(
        name=payload.name,
        age=payload.age,
        gender=payload.gender,
        known_conditions=payload.known_conditions,
        symptoms=payload.symptoms,
        allergies=payload.allergies,
        current_medications=[med.model_dump() for med in (payload.current_medications or [])],
        supplements=[supp.model_dump() for supp in (payload.supplements or [])],
        preferred_interventions=payload.preferred_interventions,
        emergency_contact=payload.emergency_contact,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def update_user_profile(db: Session, payload: UserProfileUpdate) -> Optional[UserProfile]:
    """Update the user profile."""
    obj = get_user_profile(db)
    if not obj:
        return None

    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        if k in ["current_medications", "supplements"] and v is not None:
            # Convert MedicationItem objects to dicts
            setattr(obj, k, [item.model_dump() if hasattr(item, 'model_dump') else item for item in v])
        else:
            setattr(obj, k, v)

    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def user_profile_to_dict(profile: UserProfile) -> dict:
    """Convert UserProfile to dictionary."""
    return {
        "id": profile.id,
        "name": profile.name,
        "age": profile.age,
        "gender": profile.gender,
        "known_conditions": profile.known_conditions or [],
        "symptoms": profile.symptoms or [],
        "allergies": profile.allergies or [],
        "current_medications": profile.current_medications or [],
        "supplements": profile.supplements or [],
        "preferred_interventions": profile.preferred_interventions or [],
        "emergency_contact": profile.emergency_contact,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }


# Routine CRUD helpers
def create_routine(db: Session, payload: RoutineCreate) -> Routine:
    """Create a new routine entry."""
    obj = Routine(
        entry_id=payload.entry_id,
        routine_type=payload.routine_type,
        start_time=payload.start_time,
        end_time=payload.end_time,
        duration_minutes=payload.duration_minutes,
        activities=payload.activities,
        notes=payload.notes,
        symptoms_improved=payload.symptoms_improved,
        emotions_improved=payload.emotions_improved,
        sensations_improved=payload.sensations_improved,
        overall_effectiveness=payload.overall_effectiveness,
        energy_level_after=payload.energy_level_after,
        mood_improvement=payload.mood_improvement,
        symptom_relief=payload.symptom_relief,
        completed=payload.completed,
        completion_percentage=payload.completion_percentage,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_routines_by_entry(db: Session, entry_id: int) -> List[Routine]:
    """Get all routines for a specific entry."""
    return db.query(Routine).filter(Routine.entry_id == entry_id).order_by(Routine.start_time).all()


def get_routine(db: Session, routine_id: int) -> Optional[Routine]:
    """Get a specific routine by ID."""
    return db.get(Routine, routine_id)


def update_routine(db: Session, routine_id: int, payload: RoutineUpdate) -> Optional[Routine]:
    """Update a routine."""
    obj = db.get(Routine, routine_id)
    if not obj:
        return None

    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(obj, k, v)

    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def delete_routine(db: Session, routine_id: int) -> bool:
    """Delete a routine."""
    obj = db.get(Routine, routine_id)
    if not obj:
        return False
    db.delete(obj)
    db.commit()
    return True


def routine_to_dict(routine: Routine) -> dict:
    """Convert Routine to dictionary."""
    return {
        "id": routine.id,
        "entry_id": routine.entry_id,
        "routine_type": routine.routine_type,
        "start_time": routine.start_time,
        "end_time": routine.end_time,
        "duration_minutes": routine.duration_minutes,
        "activities": routine.activities or [],
        "notes": routine.notes,
        "symptoms_improved": routine.symptoms_improved or [],
        "emotions_improved": routine.emotions_improved or [],
        "sensations_improved": routine.sensations_improved or [],
        "overall_effectiveness": routine.overall_effectiveness,
        "energy_level_after": routine.energy_level_after,
        "mood_improvement": routine.mood_improvement,
        "symptom_relief": routine.symptom_relief,
        "completed": routine.completed,
        "completion_percentage": routine.completion_percentage,
        "created_at": routine.created_at.isoformat() if routine.created_at else None,
        "updated_at": routine.updated_at.isoformat() if routine.updated_at else None,
    }


# Medication Reminder CRUD helpers
def create_medication_reminder(db: Session, payload: MedicationReminderCreate) -> MedicationReminder:
    """Create a new medication reminder."""
    obj = MedicationReminder(
        medication_name=payload.medication_name,
        dose=payload.dose,
        frequency=payload.frequency,
        custom_schedule=payload.custom_schedule,
        reminder_times=payload.reminder_times,
        enabled=payload.enabled,
        notification_method=payload.notification_method,
        notes=payload.notes
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_medication_reminders(db: Session, enabled_only: bool = False) -> List[MedicationReminder]:
    """Get all medication reminders."""
    query = db.query(MedicationReminder)
    if enabled_only:
        query = query.filter(MedicationReminder.enabled == True)
    return query.order_by(MedicationReminder.created_at.desc()).all()


def get_medication_reminder(db: Session, reminder_id: int) -> Optional[MedicationReminder]:
    """Get a specific medication reminder."""
    return db.get(MedicationReminder, reminder_id)


def update_medication_reminder(db: Session, reminder_id: int, payload: MedicationReminderUpdate) -> Optional[
    MedicationReminder]:
    """Update a medication reminder."""
    obj = db.get(MedicationReminder, reminder_id)
    if not obj:
        return None

    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(obj, k, v)

    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def delete_medication_reminder(db: Session, reminder_id: int) -> bool:
    """Delete a medication reminder."""
    obj = db.get(MedicationReminder, reminder_id)
    if not obj:
        return False
    db.delete(obj)
    db.commit()
    return True


def medication_reminder_to_dict(reminder: MedicationReminder) -> dict:
    """Convert MedicationReminder to dictionary."""
    return {
        "id": reminder.id,
        "medication_name": reminder.medication_name,
        "dose": reminder.dose,
        "frequency": reminder.frequency,
        "custom_schedule": reminder.custom_schedule,
        "reminder_times": reminder.reminder_times,
        "enabled": reminder.enabled,
        "notification_method": reminder.notification_method,
        "notes": reminder.notes,
        "last_taken": reminder.last_taken.isoformat() if reminder.last_taken else None,
        "next_reminder": reminder.next_reminder.isoformat() if reminder.next_reminder else None,
        "missed_doses": reminder.missed_doses,
        "created_at": reminder.created_at.isoformat() if reminder.created_at else None,
        "updated_at": reminder.updated_at.isoformat() if reminder.updated_at else None,
    }


# Weather Data CRUD helpers
def create_weather_data(db: Session, payload: WeatherDataCreate) -> WeatherData:
    """Create new weather data."""
    obj = WeatherData(
        date=payload.date,
        temperature_high=payload.temperature_high,
        temperature_low=payload.temperature_low,
        humidity=payload.humidity,
        pressure=payload.pressure,
        wind_speed=payload.wind_speed,
        precipitation=payload.precipitation,
        weather_condition=payload.weather_condition,
        location=payload.location,
        latitude=payload.latitude,
        longitude=payload.longitude,
        symptom_correlation=payload.symptom_correlation
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_weather_data(db: Session, start: Optional[date] = None, end: Optional[date] = None) -> List[WeatherData]:
    """Get weather data for a date range."""
    query = db.query(WeatherData)
    if start:
        query = query.filter(WeatherData.date >= start)
    if end:
        query = query.filter(WeatherData.date <= end)
    return query.order_by(WeatherData.date.desc()).all()


def weather_data_to_dict(weather: WeatherData) -> dict:
    """Convert WeatherData to dictionary."""
    return {
        "id": weather.id,
        "date": weather.date.isoformat(),
        "temperature_high": weather.temperature_high,
        "temperature_low": weather.temperature_low,
        "humidity": weather.humidity,
        "pressure": weather.pressure,
        "wind_speed": weather.wind_speed,
        "precipitation": weather.precipitation,
        "weather_condition": weather.weather_condition,
        "location": weather.location,
        "latitude": weather.latitude,
        "longitude": weather.longitude,
        "symptom_correlation": weather.symptom_correlation,
        "created_at": weather.created_at.isoformat() if weather.created_at else None,
    }


# Emergency Alert CRUD helpers
def create_emergency_alert(db: Session, payload: EmergencyAlertCreate) -> EmergencyAlert:
    """Create a new emergency alert."""
    obj = EmergencyAlert(
        alert_type=payload.alert_type,
        severity=payload.severity,
        triggered_by=payload.triggered_by,
        trigger_data=payload.trigger_data,
        user_notes=payload.user_notes,
        status=payload.status,
        response_action=payload.response_action,
        contacted_emergency_services=payload.contacted_emergency_services
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_emergency_alerts(db: Session, status: Optional[str] = None) -> List[EmergencyAlert]:
    """Get emergency alerts."""
    query = db.query(EmergencyAlert)
    if status:
        query = query.filter(EmergencyAlert.status == status)
    return query.order_by(EmergencyAlert.created_at.desc()).all()


def update_emergency_alert(db: Session, alert_id: int, payload: EmergencyAlertUpdate) -> Optional[EmergencyAlert]:
    """Update an emergency alert."""
    obj = db.get(EmergencyAlert, alert_id)
    if not obj:
        return None

    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(obj, k, v)

    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def emergency_alert_to_dict(alert: EmergencyAlert) -> dict:
    """Convert EmergencyAlert to dictionary."""
    return {
        "id": alert.id,
        "alert_type": alert.alert_type,
        "severity": alert.severity,
        "triggered_by": alert.triggered_by,
        "trigger_data": alert.trigger_data,
        "user_notes": alert.user_notes,
        "status": alert.status,
        "response_action": alert.response_action,
        "contacted_emergency_services": alert.contacted_emergency_services,
        "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
        "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
        "created_at": alert.created_at.isoformat() if alert.created_at else None,
        "updated_at": alert.updated_at.isoformat() if alert.updated_at else None,
    }


# Chat Session CRUD helpers
def create_chat_session(db: Session, session_id: str) -> ChatSession:
    """Create a new chat session."""
    obj = ChatSession(session_id=session_id)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_chat_session(db: Session, session_id: str) -> Optional[ChatSession]:
    """Get a chat session by session ID."""
    return db.query(ChatSession).filter(ChatSession.session_id == session_id).first()


def update_chat_session(db: Session, session_id: str, user_context: Optional[Dict] = None,
                        conversation_history: Optional[List] = None) -> Optional[ChatSession]:
    """Update a chat session."""
    obj = get_chat_session(db, session_id)
    if not obj:
        return None

    if user_context is not None:
        obj.user_context = user_context
    if conversation_history is not None:
        obj.conversation_history = conversation_history

    obj.last_activity = datetime.now()
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def chat_session_to_dict(session: ChatSession) -> dict:
    """Convert ChatSession to dictionary."""
    return {
        "id": session.id,
        "session_id": session.session_id,
        "user_context": session.user_context,
        "conversation_history": session.conversation_history,
        "active": session.active,
        "last_activity": session.last_activity.isoformat() if session.last_activity else None,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
    }


# ---------------------------
# Analytics
# ---------------------------
NUMERIC_COLS = [
    "sleep_hours",
    "sleep_quality",
    "mood_valence",
    "anxiety_level",
    "depression_level",
    "mania_level",
    "adhd_focus",
    "productivity_score",
    "hrv",
]


def to_dataframe(entries: List[Entry]) -> pd.DataFrame:
    records = [entry_to_dict(e) for e in entries]
    return pd.DataFrame.from_records(records)


def summary(entries: List[Entry]) -> dict:
    if not entries:
        return {"count": 0, "averages": {}, "migraine_days": 0}
    df = to_dataframe(entries)
    df["migraine_bin"] = df["migraine"].astype(int)
    averages = {c: float(df[c].dropna().mean()) for c in NUMERIC_COLS if c in df.columns}
    migraine_days = int(df["migraine_bin"].sum())
    return {"count": int(len(df)), "averages": averages, "migraine_days": migraine_days}


def trigger_risk(entries: List[Entry], today: date) -> dict:
    """Heuristic:
    If (yesterday.sleep_hours < X) AND (yesterday.anxiety_level >= Y) => 'high' risk today.
    Else 'baseline'. If no yesterday data => 'unknown'.
    """
    y = today - timedelta(days=1)
    e_y = next((e for e in entries if e.date == y), None)
    if not e_y:
        return {"date": today.isoformat(), "risk": "unknown", "reason": "no_yesterday_data"}
    sh = e_y.sleep_hours or 0.0
    anx = e_y.anxiety_level or 0
    if sh < TRIGGER_SLEEP_HOURS_LT and anx >= TRIGGER_ANXIETY_GE:
        return {"date": today.isoformat(), "risk": "high",
                "reason": f"sleep_hours<{TRIGGER_SLEEP_HOURS_LT} and anxiety>={TRIGGER_ANXIETY_GE} yesterday"}
    return {"date": today.isoformat(), "risk": "baseline", "reason": "conditions_not_met"}


def routines_scoring(entries: List[Entry]) -> dict:
    """Per-day habit score (normalized count) and per-routine current/best streaks."""
    routine_set = set()
    for e in entries:
        for x in (e.routines_followed or []):
            routine_set.add(x)
    routines = sorted(list(routine_set))

    current_streak = {r: 0 for r in routines}
    best_streak = {r: 0 for r in routines}
    prev_day_had = {r: False for r in routines}

    # entries should be in ascending date order
    for e in entries:
        today = set(e.routines_followed or [])
        for r in routines:
            if r in today:
                if prev_day_had[r]:
                    current_streak[r] += 1
                else:
                    current_streak[r] = 1
                best_streak[r] = max(best_streak[r], current_streak[r])
                prev_day_had[r] = True
            else:
                prev_day_had[r] = False
                current_streak[r] = 0

    # Daily habit score
    max_routines_any_day = 0
    for e in entries:
        max_routines_any_day = max(max_routines_any_day, len(e.routines_followed or []))
    per_day_scores = []
    for e in entries:
        cnt = len(e.routines_followed or [])
        score = 0.0 if max_routines_any_day == 0 else round(cnt / max_routines_any_day, 3)
        per_day_scores.append({"date": e.date.isoformat(), "score": score, "count": cnt})

    return {
        "routines": routines,
        "best_streaks": best_streak,
        "current_streaks": current_streak,
        "daily_scores": per_day_scores,
    }


def calendar_heatmap_points(entries: List[Entry], start: date, end: date) -> List[dict]:
    """Return list of {date, value} using migraine_intensity (0 if None)."""
    out = []
    for e in entries:
        if start <= e.date <= end:
            out.append({"date": e.date.isoformat(), "value": int(e.migraine_intensity or 0)})
    return out


def body_map_analytics(entries: List[Entry]) -> dict:
    """Analyze body sensations across all entries."""
    if not entries:
        return {"total_entries": 0, "body_parts": {}, "most_common": []}

    # Aggregate sensations by body part
    body_parts = {}
    total_entries = 0

    for entry in entries:
        if entry.body_sensations:
            total_entries += 1
            for body_part, intensity in entry.body_sensations.items():
                if body_part not in body_parts:
                    body_parts[body_part] = {
                        "total_intensity": 0,
                        "count": 0,
                        "max_intensity": 0,
                        "avg_intensity": 0.0
                    }
                body_parts[body_part]["total_intensity"] += intensity
                body_parts[body_part]["count"] += 1
                body_parts[body_part]["max_intensity"] = max(
                    body_parts[body_part]["max_intensity"], intensity
                )

    # Calculate averages
    for body_part in body_parts:
        count = body_parts[body_part]["count"]
        if count > 0:
            body_parts[body_part]["avg_intensity"] = round(
                body_parts[body_part]["total_intensity"] / count, 2
            )

    # Find most common body parts
    most_common = sorted(
        body_parts.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )[:10]

    return {
        "total_entries": total_entries,
        "body_parts": body_parts,
        "most_common": [{"part": part, **stats} for part, stats in most_common]
    }


def body_map_heatmap_data(entries: List[Entry], start: date, end: date) -> dict:
    """Generate heatmap data for body map visualization."""
    # Define body regions with their coordinates for visualization
    body_regions = {
        "head": {"x": 0.5, "y": 0.1, "radius": 0.08},
        "neck": {"x": 0.5, "y": 0.2, "radius": 0.06},
        "shoulders": {"x": 0.5, "y": 0.25, "radius": 0.12},
        "chest": {"x": 0.5, "y": 0.35, "radius": 0.1},
        "upper_back": {"x": 0.5, "y": 0.35, "radius": 0.1},
        "arms": {"x": 0.3, "y": 0.4, "radius": 0.05},
        "hands": {"x": 0.25, "y": 0.6, "radius": 0.04},
        "stomach": {"x": 0.5, "y": 0.5, "radius": 0.08},
        "lower_back": {"x": 0.5, "y": 0.5, "radius": 0.08},
        "hips": {"x": 0.5, "y": 0.65, "radius": 0.1},
        "legs": {"x": 0.5, "y": 0.8, "radius": 0.08},
        "feet": {"x": 0.5, "y": 0.95, "radius": 0.06}
    }

    # Aggregate sensations for the date range
    region_intensities = {region: [] for region in body_regions}

    for entry in entries:
        if start <= entry.date <= end and entry.body_sensations:
            for body_part, intensity in entry.body_sensations.items():
                # Map body parts to regions
                for region in body_regions:
                    if body_part.lower() in region or region in body_part.lower():
                        region_intensities[region].append(intensity)
                        break

    # Calculate average intensity for each region
    heatmap_data = {}
    for region, intensities in region_intensities.items():
        if intensities:
            avg_intensity = sum(intensities) / len(intensities)
            heatmap_data[region] = {
                **body_regions[region],
                "intensity": round(avg_intensity, 2),
                "count": len(intensities)
            }
        else:
            heatmap_data[region] = {
                **body_regions[region],
                "intensity": 0,
                "count": 0
            }

    return heatmap_data


def get_intervention_recommendations(body_sensations: Dict[str, int], entries: List[Entry],
                                     user_profile: Optional[UserProfile] = None) -> dict:
    """Generate personalized intervention recommendations based on body sensations, historical data, and user profile."""

    # Define body region to intervention mappings
    body_interventions = {
        "head": {
            "high_intensity": ["meditation", "deep_breathing", "headache_medication", "rest_in_dark_room"],
            "medium_intensity": ["meditation", "gentle_exercise", "hydration", "eye_rest"],
            "low_intensity": ["light_walk", "fresh_air", "mindful_breathing"]
        },
        "neck": {
            "high_intensity": ["neck_stretches", "heat_therapy", "massage", "posture_correction"],
            "medium_intensity": ["gentle_stretching", "ergonomic_adjustment", "stress_reduction"],
            "low_intensity": ["posture_awareness", "light_movement"]
        },
        "shoulders": {
            "high_intensity": ["shoulder_stretches", "massage", "stress_management", "exercise"],
            "medium_intensity": ["stretching", "relaxation_techniques", "posture_work"],
            "low_intensity": ["light_movement", "awareness_practice"]
        },
        "chest": {
            "high_intensity": ["deep_breathing", "anxiety_medication", "calming_techniques", "professional_help"],
            "medium_intensity": ["breathing_exercises", "meditation", "social_connection"],
            "low_intensity": ["mindful_breathing", "gentle_activity"]
        },
        "stomach": {
            "high_intensity": ["dietary_adjustment", "digestive_medication", "stress_reduction", "rest"],
            "medium_intensity": ["gentle_walking", "mindful_eating", "relaxation"],
            "low_intensity": ["light_activity", "hydration"]
        },
        "arms": {
            "high_intensity": ["arm_stretches", "massage", "rest", "ergonomic_adjustment"],
            "medium_intensity": ["gentle_stretching", "movement", "stress_reduction"],
            "low_intensity": ["light_exercise", "awareness"]
        },
        "hands": {
            "high_intensity": ["hand_stretches", "rest", "ergonomic_adjustment", "stress_reduction"],
            "medium_intensity": ["gentle_movement", "awareness", "breaks"],
            "low_intensity": ["light_activity", "mindfulness"]
        },
        "legs": {
            "high_intensity": ["leg_stretches", "massage", "rest", "gentle_walking"],
            "medium_intensity": ["stretching", "light_exercise", "movement"],
            "low_intensity": ["walking", "gentle_activity"]
        },
        "feet": {
            "high_intensity": ["foot_stretches", "massage", "rest", "proper_footwear"],
            "medium_intensity": ["gentle_movement", "stretching", "comfort"],
            "low_intensity": ["light_walking", "awareness"]
        }
    }

    # Analyze historical effectiveness
    intervention_effectiveness = analyze_intervention_effectiveness(entries)

    # Generate recommendations for each body part
    recommendations = {}
    overall_recommendations = []

    for body_part, intensity in body_sensations.items():
        if intensity > 0:
            # Determine intensity level
            if intensity >= 7:
                intensity_level = "high_intensity"
            elif intensity >= 4:
                intensity_level = "medium_intensity"
            else:
                intensity_level = "low_intensity"

            # Get base interventions for this body part and intensity
            base_interventions = body_interventions.get(body_part.lower(), {}).get(intensity_level, [])

            # Prioritize based on historical effectiveness
            prioritized_interventions = prioritize_interventions(base_interventions, intervention_effectiveness)

            recommendations[body_part] = {
                "intensity": intensity,
                "interventions": prioritized_interventions[:5],  # Top 5 recommendations
                "urgency": "high" if intensity >= 7 else "medium" if intensity >= 4 else "low"
            }

            overall_recommendations.extend(prioritized_interventions[:3])

    # Get overall top recommendations
    overall_recommendations = list(set(overall_recommendations))  # Remove duplicates
    overall_recommendations = prioritize_interventions(overall_recommendations, intervention_effectiveness)

    # Add medication suggestions if applicable
    medication_suggestions = get_medication_suggestions(body_sensations, entries, user_profile)

    # Personalize recommendations based on user profile
    if user_profile:
        # Prioritize user's preferred interventions
        if user_profile.preferred_interventions:
            overall_recommendations = prioritize_user_preferences(overall_recommendations,
                                                                  user_profile.preferred_interventions)

        # Add condition-specific recommendations
        condition_recommendations = get_condition_specific_recommendations(user_profile, body_sensations)

        return {
            "body_part_recommendations": recommendations,
            "overall_recommendations": overall_recommendations[:5],
            "medication_suggestions": medication_suggestions,
            "condition_recommendations": condition_recommendations,
            "urgency_level": "high" if any(
                r["urgency"] == "high" for r in recommendations.values()) else "medium" if any(
                r["urgency"] == "medium" for r in recommendations.values()) else "low"
        }

    return {
        "body_part_recommendations": recommendations,
        "overall_recommendations": overall_recommendations[:5],
        "medication_suggestions": medication_suggestions,
        "urgency_level": "high" if any(r["urgency"] == "high" for r in recommendations.values()) else "medium" if any(
            r["urgency"] == "medium" for r in recommendations.values()) else "low"
    }


def analyze_intervention_effectiveness(entries: List[Entry]) -> dict:
    """Analyze which interventions have been most effective based on historical data."""
    if not entries:
        return {}

    # Track intervention effectiveness (simplified heuristic)
    effectiveness = {
        "meditation": 0.8,
        "deep_breathing": 0.7,
        "exercise": 0.9,
        "walking": 0.8,
        "stretching": 0.7,
        "massage": 0.8,
        "social_connection": 0.6,
        "journaling": 0.5,
        "prayer": 0.6,
        "rest": 0.7,
        "hydration": 0.6,
        "fresh_air": 0.7,
        "mindful_breathing": 0.7,
        "stress_reduction": 0.8,
        "gentle_exercise": 0.8,
        "light_walk": 0.7,
        "eye_rest": 0.6,
        "posture_correction": 0.7,
        "ergonomic_adjustment": 0.8,
        "calming_techniques": 0.8,
        "dietary_adjustment": 0.6,
        "mindful_eating": 0.5,
        "gentle_movement": 0.7,
        "awareness_practice": 0.6,
        "breaks": 0.7,
        "mindfulness": 0.7,
        "light_activity": 0.7,
        "gentle_activity": 0.7,
        "comfort": 0.6,
        "awareness": 0.5
    }

    # Adjust based on user's historical patterns
    for entry in entries[-30:]:  # Last 30 entries
        if entry.body_sensations:
            # If user has high sensations but good mood/productivity, interventions might be working
            avg_sensation = sum(entry.body_sensations.values()) / len(
                entry.body_sensations) if entry.body_sensations else 0
            mood_score = entry.mood_valence or 0
            productivity = entry.productivity_score or 50

            if avg_sensation > 5 and mood_score > 0 and productivity > 60:
                # User is managing well despite sensations
                for intervention in effectiveness:
                    effectiveness[intervention] = min(1.0, effectiveness[intervention] + 0.05)

    return effectiveness


def prioritize_interventions(interventions: List[str], effectiveness: dict) -> List[str]:
    """Prioritize interventions based on effectiveness scores."""
    if not interventions:
        return []

    # Sort by effectiveness score
    prioritized = sorted(interventions, key=lambda x: effectiveness.get(x, 0.5), reverse=True)
    return prioritized


def get_medication_suggestions(body_sensations: Dict[str, int], entries: List[Entry],
                               user_profile: Optional[UserProfile] = None) -> List[dict]:
    """Suggest medications based on body sensations, historical medication use, and user profile."""
    suggestions = []

    # Analyze historical medication effectiveness
    med_effectiveness = {}
    for entry in entries[-50:]:  # Last 50 entries
        if entry.meds_cipher:
            meds = _deserialize_meds(entry.meds_cipher)
            if meds:
                for med in meds:
                    med_name = med.get("name", "").lower()
                    if med_name not in med_effectiveness:
                        med_effectiveness[med_name] = {"count": 0, "total_intensity": 0}

                    # Calculate average body sensation intensity for this medication
                    if entry.body_sensations:
                        avg_intensity = sum(entry.body_sensations.values()) / len(entry.body_sensations)
                        med_effectiveness[med_name]["total_intensity"] += avg_intensity
                        med_effectiveness[med_name]["count"] += 1

    # Generate suggestions based on body parts
    for body_part, intensity in body_sensations.items():
        if intensity >= 6:  # Only suggest for moderate-high intensity
            if "head" in body_part.lower():
                suggestions.append({
                    "type": "pain_relief",
                    "suggestions": ["acetaminophen", "ibuprofen", "aspirin"],
                    "body_part": body_part,
                    "intensity": intensity,
                    "reason": "Head pain detected"
                })
            elif "chest" in body_part.lower() and intensity >= 7:
                suggestions.append({
                    "type": "anxiety_relief",
                    "suggestions": ["consult_doctor_for_anxiety_medication"],
                    "body_part": body_part,
                    "intensity": intensity,
                    "reason": "High chest tension - consider professional consultation"
                })
            elif "stomach" in body_part.lower():
                suggestions.append({
                    "type": "digestive_relief",
                    "suggestions": ["antacids", "pepto_bismol", "consult_doctor"],
                    "body_part": body_part,
                    "intensity": intensity,
                    "reason": "Stomach discomfort detected"
                })

    # Add historically effective medications
    effective_meds = []
    for med_name, stats in med_effectiveness.items():
        if stats["count"] >= 3:  # Used at least 3 times
            avg_effectiveness = 10 - (stats["total_intensity"] / stats["count"])  # Lower intensity = more effective
            if avg_effectiveness > 5:  # Effective threshold
                effective_meds.append({
                    "name": med_name,
                    "effectiveness_score": round(avg_effectiveness, 2),
                    "usage_count": stats["count"]
                })

    if effective_meds:
        suggestions.append({
            "type": "historically_effective",
            "suggestions": [med["name"] for med in
                            sorted(effective_meds, key=lambda x: x["effectiveness_score"], reverse=True)[:3]],
            "reason": "Based on your historical data"
        })

    # Add current medications from user profile
    if user_profile and user_profile.current_medications:
        current_meds = [med.get("name", "") for med in user_profile.current_medications if med.get("name")]
        if current_meds:
            suggestions.append({
                "type": "current_medications",
                "suggestions": current_meds,
                "reason": "Your current medications"
            })

    return suggestions


def prioritize_user_preferences(interventions: List[str], preferred_interventions: List[str]) -> List[str]:
    """Prioritize interventions based on user preferences."""
    if not preferred_interventions:
        return interventions

    # Move preferred interventions to the front
    prioritized = []
    for pref in preferred_interventions:
        if pref in interventions:
            prioritized.append(pref)
            interventions.remove(pref)

    # Add remaining interventions
    prioritized.extend(interventions)
    return prioritized


def get_condition_specific_recommendations(user_profile: UserProfile, body_sensations: Dict[str, int]) -> List[dict]:
    """Get recommendations specific to user's known conditions."""
    recommendations = []

    if not user_profile.known_conditions:
        return recommendations

    # Condition-specific intervention mappings
    condition_interventions = {
        "migraine": ["rest_in_dark_room", "hydration", "caffeine", "prescription_migraine_medication"],
        "anxiety": ["deep_breathing", "meditation", "progressive_muscle_relaxation", "anxiety_medication"],
        "depression": ["exercise", "social_connection", "sunlight_exposure", "therapy"],
        "adhd": ["focus_techniques", "exercise", "meditation", "adhd_medication"],
        "insomnia": ["sleep_hygiene", "relaxation_techniques", "sleep_medication", "bedtime_routine"],
        "chronic_pain": ["gentle_stretching", "heat_therapy", "pain_medication", "physical_therapy"],
        "digestive_issues": ["dietary_adjustment", "probiotics", "digestive_medication", "stress_reduction"]
    }

    for condition in user_profile.known_conditions:
        condition_lower = condition.lower()
        for cond_key, interventions in condition_interventions.items():
            if cond_key in condition_lower or condition_lower in cond_key:
                recommendations.append({
                    "condition": condition,
                    "interventions": interventions,
                    "reason": f"Based on your {condition} diagnosis"
                })
                break

    return recommendations


def analyze_routine_effectiveness(routines: List[Routine]) -> dict:
    """Analyze routine effectiveness across different types and metrics."""
    if not routines:
        return {"total_routines": 0, "routine_types": {}, "most_effective": [], "improvement_areas": []}

    # Group routines by type
    routine_types = {}
    for routine in routines:
        if routine.routine_type not in routine_types:
            routine_types[routine.routine_type] = {
                "count": 0,
                "total_effectiveness": 0,
                "total_energy": 0,
                "total_mood": 0,
                "total_symptom_relief": 0,
                "avg_duration": 0,
                "completion_rate": 0,
                "activities": {},
                "symptoms_improved": {},
                "emotions_improved": {},
                "sensations_improved": {}
            }

        rt = routine_types[routine.routine_type]
        rt["count"] += 1

        # Aggregate effectiveness scores
        if routine.overall_effectiveness is not None:
            rt["total_effectiveness"] += routine.overall_effectiveness
        if routine.energy_level_after is not None:
            rt["total_energy"] += routine.energy_level_after
        if routine.mood_improvement is not None:
            rt["total_mood"] += routine.mood_improvement
        if routine.symptom_relief is not None:
            rt["total_symptom_relief"] += routine.symptom_relief

        # Track duration
        if routine.duration_minutes:
            rt["avg_duration"] += routine.duration_minutes

        # Track completion
        if routine.completed:
            rt["completion_rate"] += 1

        # Track activities
        if routine.activities:
            for activity in routine.activities:
                rt["activities"][activity] = rt["activities"].get(activity, 0) + 1

        # Track improvements
        if routine.symptoms_improved:
            for symptom in routine.symptoms_improved:
                rt["symptoms_improved"][symptom] = rt["symptoms_improved"].get(symptom, 0) + 1

        if routine.emotions_improved:
            for emotion in routine.emotions_improved:
                rt["emotions_improved"][emotion] = rt["emotions_improved"].get(emotion, 0) + 1

        if routine.sensations_improved:
            for sensation in routine.sensations_improved:
                rt["sensations_improved"][sensation] = rt["sensations_improved"].get(sensation, 0) + 1

    # Calculate averages and percentages
    for rt_type, stats in routine_types.items():
        count = stats["count"]
        if count > 0:
            stats["avg_effectiveness"] = round(stats["total_effectiveness"] / count, 2)
            stats["avg_energy"] = round(stats["total_energy"] / count, 2)
            stats["avg_mood"] = round(stats["total_mood"] / count, 2)
            stats["avg_symptom_relief"] = round(stats["total_symptom_relief"] / count, 2)
            stats["avg_duration"] = round(stats["avg_duration"] / count, 2)
            stats["completion_rate"] = round((stats["completion_rate"] / count) * 100, 1)

    # Find most effective routine types
    most_effective = sorted(
        routine_types.items(),
        key=lambda x: x[1]["avg_effectiveness"],
        reverse=True
    )

    # Find improvement areas (lowest effectiveness)
    improvement_areas = sorted(
        routine_types.items(),
        key=lambda x: x[1]["avg_effectiveness"]
    )

    return {
        "total_routines": len(routines),
        "routine_types": routine_types,
        "most_effective": [{"type": rt_type, **stats} for rt_type, stats in most_effective],
        "improvement_areas": [{"type": rt_type, **stats} for rt_type, stats in improvement_areas]
    }


def get_routine_recommendations(user_profile: Optional[UserProfile] = None, routines: List[Routine] = None) -> dict:
    """Generate routine recommendations based on effectiveness analysis."""
    if not routines:
        return {"recommendations": [], "suggested_improvements": []}

    analysis = analyze_routine_effectiveness(routines)

    recommendations = []
    suggested_improvements = []

    # Generate recommendations based on most effective routines
    for effective in analysis["most_effective"][:3]:  # Top 3
        if effective["avg_effectiveness"] >= 7:
            recommendations.append({
                "routine_type": effective["type"],
                "reason": f"High effectiveness score ({effective['avg_effectiveness']}/10)",
                "suggestions": [
                    "Continue this routine consistently",
                    "Consider increasing duration if beneficial",
                    "Share this routine with others"
                ]
            })

    # Generate improvement suggestions
    for improvement in analysis["improvement_areas"][:3]:  # Bottom 3
        if improvement["avg_effectiveness"] < 6:
            suggested_improvements.append({
                "routine_type": improvement["type"],
                "current_score": improvement["avg_effectiveness"],
                "suggestions": [
                    "Try different activities within this routine",
                    "Adjust timing or duration",
                    "Consider breaking into smaller segments",
                    "Add more enjoyable activities"
                ]
            })

    # Add personalized recommendations based on user profile
    if user_profile and user_profile.known_conditions:
        for condition in user_profile.known_conditions:
            condition_lower = condition.lower()

            # Condition-specific routine suggestions
            if "anxiety" in condition_lower:
                recommendations.append({
                    "routine_type": "morning",
                    "reason": "Based on your anxiety diagnosis",
                    "suggestions": [
                        "Add 10-15 minutes of meditation",
                        "Practice deep breathing exercises",
                        "Create a calming morning environment"
                    ]
                })
            elif "depression" in condition_lower:
                recommendations.append({
                    "routine_type": "exercise",
                    "reason": "Based on your depression diagnosis",
                    "suggestions": [
                        "Start with light walking or stretching",
                        "Exercise outdoors for sunlight exposure",
                        "Set small, achievable goals"
                    ]
                })
            elif "adhd" in condition_lower:
                recommendations.append({
                    "routine_type": "work",
                    "reason": "Based on your ADHD diagnosis",
                    "suggestions": [
                        "Use time-blocking techniques",
                        "Take regular short breaks",
                        "Create a distraction-free environment"
                    ]
                })

    return {
        "recommendations": recommendations,
        "suggested_improvements": suggested_improvements,
        "effectiveness_analysis": analysis
    }


# ---------------------------
# Enhanced Calendar and Streak Functions
# ---------------------------
def get_logging_streak(entries: List[Entry]) -> dict:
    """Calculate consecutive logging days and streak milestones."""
    if not entries:
        return {
            "current_streak": 0,
            "best_streak": 0,
            "total_logged_days": 0,
            "streak_active": False,
            "days_to_next_milestone": 3,
            "milestones": []
        }

    # Sort entries by date
    sorted_entries = sorted(entries, key=lambda x: x.date)
    dates = [e.date for e in sorted_entries]

    # Calculate current streak (consecutive days from most recent)
    current_streak = 0
    best_streak = 0
    temp_streak = 0
    today = date.today()

    # Start from the most recent entry and work backwards
    for i in range(len(dates) - 1, -1, -1):
        if i == len(dates) - 1:
            # Most recent entry
            if dates[i] == today or dates[i] == today - timedelta(days=1):
                current_streak = 1
                temp_streak = 1
            else:
                current_streak = 0
                temp_streak = 0
        else:
            # Check if consecutive
            if (dates[i + 1] - dates[i]).days == 1:
                temp_streak += 1
                if i == len(dates) - 2:  # Second most recent
                    current_streak = temp_streak
            else:
                temp_streak = 0

        best_streak = max(best_streak, temp_streak)

    # Calculate milestones
    milestones = [3, 7, 14, 30, 60, 90, 180, 365]
    days_to_next_milestone = None

    for milestone in milestones:
        if current_streak < milestone:
            days_to_next_milestone = milestone - current_streak
            break

    # Determine if streak is active (3+ days)
    streak_active = current_streak >= 3

    return {
        "current_streak": current_streak,
        "best_streak": best_streak,
        "total_logged_days": len(dates),
        "streak_active": streak_active,
        "days_to_next_milestone": days_to_next_milestone,
        "milestones": milestones,
        "last_logged_date": dates[-1].isoformat() if dates else None
    }


def get_calendar_data(entries: List[Entry], routines: List[Routine], start: date, end: date) -> dict:
    """Get comprehensive calendar data including entries, routines, and streaks."""
    # Create date range
    date_range = []
    current = start
    while current <= end:
        date_range.append(current)
        current += timedelta(days=1)

    # Organize entries by date
    entries_by_date = {}
    for entry in entries:
        if start <= entry.date <= end:
            entries_by_date[entry.date] = {
                "entry": entry,
                "has_entry": True,
                "migraine": entry.migraine,
                "migraine_intensity": entry.migraine_intensity or 0,
                "mood": entry.mood_valence,
                "anxiety": entry.anxiety_level,
                "sleep_hours": entry.sleep_hours,
                "body_sensations": entry.body_sensations or {},
                "routines_followed": entry.routines_followed or []
            }

    # Organize routines by date
    routines_by_date = {}
    for routine in routines:
        if hasattr(routine, 'entry_id') and routine.entry_id:
            # Find the entry for this routine
            for entry in entries:
                if entry.id == routine.entry_id:
                    if entry.date not in routines_by_date:
                        routines_by_date[entry.date] = []
                    routines_by_date[entry.date].append({
                        "id": routine.id,
                        "type": routine.routine_type,
                        "duration": routine.duration_minutes,
                        "effectiveness": routine.overall_effectiveness,
                        "completed": routine.completed,
                        "activities": routine.activities or []
                    })
                    break

    # Build calendar data
    calendar_data = []
    for current_date in date_range:
        day_data = {
            "date": current_date.isoformat(),
            "has_entry": False,
            "has_routines": False,
            "entry_data": None,
            "routines": [],
            "summary": {
                "mood": None,
                "anxiety": None,
                "migraine": False,
                "sleep_hours": None,
                "body_sensations_count": 0,
                "routines_count": 0
            }
        }

        if current_date in entries_by_date:
            entry_data = entries_by_date[current_date]
            day_data["has_entry"] = True
            day_data["entry_data"] = entry_data
            day_data["summary"]["mood"] = entry_data["mood"]
            day_data["summary"]["anxiety"] = entry_data["anxiety"]
            day_data["summary"]["migraine"] = entry_data["migraine"]
            day_data["summary"]["sleep_hours"] = entry_data["sleep_hours"]
            day_data["summary"]["body_sensations_count"] = len(entry_data["body_sensations"])

        if current_date in routines_by_date:
            day_data["has_routines"] = True
            day_data["routines"] = routines_by_date[current_date]
            day_data["summary"]["routines_count"] = len(routines_by_date[current_date])

        calendar_data.append(day_data)

    # Calculate streak info
    streak_info = get_logging_streak(entries)

    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "calendar_data": calendar_data,
        "streak_info": streak_info,
        "total_entries": len(entries_by_date),
        "total_routines": len(routines)
    }


def get_retroactive_logging_suggestions(entries: List[Entry], target_date: date) -> dict:
    """Generate suggestions for retroactive logging based on patterns."""
    if not entries:
        return {"suggestions": [], "patterns": {}}

    # Find entries around the target date
    nearby_entries = []
    for entry in entries:
        days_diff = abs((entry.date - target_date).days)
        if days_diff <= 7:  # Within a week
            nearby_entries.append(entry)

    if not nearby_entries:
        return {"suggestions": [], "patterns": {}}

    # Analyze patterns
    patterns = {
        "avg_sleep_hours": 0,
        "avg_mood": 0,
        "avg_anxiety": 0,
        "common_routines": [],
        "common_body_sensations": {},
        "migraine_frequency": 0
    }

    total_entries = len(nearby_entries)
    total_sleep = sum(e.sleep_hours or 0 for e in nearby_entries)
    total_mood = sum(e.mood_valence or 0 for e in nearby_entries)
    total_anxiety = sum(e.anxiety_level or 0 for e in nearby_entries)
    migraine_count = sum(1 for e in nearby_entries if e.migraine)

    patterns["avg_sleep_hours"] = round(total_sleep / total_entries, 1) if total_entries > 0 else 0
    patterns["avg_mood"] = round(total_mood / total_entries, 1) if total_entries > 0 else 0
    patterns["avg_anxiety"] = round(total_anxiety / total_entries, 1) if total_entries > 0 else 0
    patterns["migraine_frequency"] = round(migraine_count / total_entries * 100, 1) if total_entries > 0 else 0

    # Common routines
    routine_counts = {}
    for entry in nearby_entries:
        for routine in entry.routines_followed or []:
            routine_counts[routine] = routine_counts.get(routine, 0) + 1

    patterns["common_routines"] = sorted(
        routine_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    # Common body sensations
    sensation_counts = {}
    for entry in nearby_entries:
        for body_part, intensity in entry.body_sensations.items():
            if body_part not in sensation_counts:
                sensation_counts[body_part] = {"count": 0, "total_intensity": 0}
            sensation_counts[body_part]["count"] += 1
            sensation_counts[body_part]["total_intensity"] += intensity

    for body_part, data in sensation_counts.items():
        patterns["common_body_sensations"][body_part] = {
            "frequency": round(data["count"] / total_entries * 100, 1),
            "avg_intensity": round(data["total_intensity"] / data["count"], 1)
        }

    # Generate suggestions
    suggestions = []

    if patterns["avg_sleep_hours"] > 0:
        suggestions.append(f"Sleep: {patterns['avg_sleep_hours']} hours (based on recent patterns)")

    if patterns["avg_mood"] != 0:
        mood_desc = "positive" if patterns["avg_mood"] > 0 else "neutral" if patterns["avg_mood"] == 0 else "negative"
        suggestions.append(f"Mood: {mood_desc} ({patterns['avg_mood']})")

    if patterns["avg_anxiety"] > 0:
        anxiety_level = "low" if patterns["avg_anxiety"] < 3 else "moderate" if patterns["avg_anxiety"] < 7 else "high"
        suggestions.append(f"Anxiety: {anxiety_level} level ({patterns['avg_anxiety']}/10)")

    if patterns["migraine_frequency"] > 20:
        suggestions.append(f"Migraine: {patterns['migraine_frequency']}% chance based on recent patterns")

    if patterns["common_routines"]:
        routine_suggestions = [f"{routine} ({count} times)" for routine, count in patterns["common_routines"][:3]]
        suggestions.append(f"Common routines: {', '.join(routine_suggestions)}")

    if patterns["common_body_sensations"]:
        sensation_suggestions = []
        for body_part, data in patterns["common_body_sensations"].items():
            if data["frequency"] > 30:  # More than 30% frequency
                sensation_suggestions.append(f"{body_part} (intensity {data['avg_intensity']})")
        if sensation_suggestions:
            suggestions.append(f"Common sensations: {', '.join(sensation_suggestions[:3])}")

    return {
        "suggestions": suggestions,
        "patterns": patterns
    }


# ---------------------------
# Advanced Features Functions
# ---------------------------

def check_emergency_alerts(entry: Entry, user_profile: Optional[UserProfile] = None) -> List[dict]:
    """Check if an entry triggers any emergency alerts."""
    alerts = []

    # High anxiety alert
    if entry.anxiety_level and entry.anxiety_level >= 9:
        alerts.append({
            "alert_type": "high_anxiety",
            "severity": "high" if entry.anxiety_level >= 9 else "medium",
            "triggered_by": f"Anxiety level: {entry.anxiety_level}/10",
            "trigger_data": {"anxiety_level": entry.anxiety_level}
        })

    # Suicidal thoughts alert (based on mood and depression)
    if (entry.mood_valence and entry.mood_valence <= -2 and
            entry.depression_level and entry.depression_level >= 8):
        alerts.append({
            "alert_type": "suicidal_thoughts",
            "severity": "critical",
            "triggered_by": f"Very negative mood ({entry.mood_valence}) and high depression ({entry.depression_level}/10)",
            "trigger_data": {
                "mood_valence": entry.mood_valence,
                "depression_level": entry.depression_level
            }
        })

    # Severe migraine alert
    if entry.migraine and entry.migraine_intensity and entry.migraine_intensity >= 8:
        alerts.append({
            "alert_type": "severe_migraine",
            "severity": "high",
            "triggered_by": f"Severe migraine (intensity: {entry.migraine_intensity}/10)",
            "trigger_data": {"migraine_intensity": entry.migraine_intensity}
        })

    # Medication overdose alert (if user has medications)
    if user_profile and user_profile.current_medications:
        # This would need more sophisticated logic based on medication tracking
        pass

    return alerts


def analyze_weather_correlation(entries: List[Entry], weather_data: List[WeatherData]) -> dict:
    """Analyze correlation between weather and symptoms."""
    if not entries or not weather_data:
        return {"correlations": {}, "insights": []}

    # Create weather lookup
    weather_lookup = {w.date: w for w in weather_data}

    correlations = {
        "temperature": {"migraine": [], "anxiety": [], "mood": []},
        "humidity": {"migraine": [], "anxiety": [], "mood": []},
        "pressure": {"migraine": [], "anxiety": [], "mood": []},
        "precipitation": {"migraine": [], "anxiety": [], "mood": []}
    }

    insights = []

    for entry in entries:
        if entry.date in weather_lookup:
            weather = weather_lookup[entry.date]

            # Temperature correlations
            if weather.temperature_high:
                if entry.migraine:
                    correlations["temperature"]["migraine"].append(weather.temperature_high)
                if entry.anxiety_level:
                    correlations["temperature"]["anxiety"].append(weather.temperature_high)
                if entry.mood_valence:
                    correlations["temperature"]["mood"].append(weather.temperature_high)

            # Humidity correlations
            if weather.humidity:
                if entry.migraine:
                    correlations["humidity"]["migraine"].append(weather.humidity)
                if entry.anxiety_level:
                    correlations["humidity"]["anxiety"].append(weather.humidity)
                if entry.mood_valence:
                    correlations["humidity"]["mood"].append(weather.humidity)

    # Generate insights
    for weather_factor, symptoms in correlations.items():
        for symptom, values in symptoms.items():
            if len(values) >= 5:  # Need at least 5 data points
                avg_value = sum(values) / len(values)
                if symptom == "migraine":
                    if weather_factor == "temperature" and avg_value > 80:
                        insights.append(f"High temperatures ({avg_value:.1f}°F) may trigger migraines")
                    elif weather_factor == "humidity" and avg_value > 70:
                        insights.append(f"High humidity ({avg_value:.1f}%) may trigger migraines")
                elif symptom == "anxiety":
                    if weather_factor == "pressure" and avg_value < 29.5:
                        insights.append(f"Low pressure ({avg_value:.1f} inHg) may increase anxiety")

    return {
        "correlations": correlations,
        "insights": insights
    }


def get_medication_reminder_schedule(reminder: MedicationReminder) -> List[datetime]:
    """Calculate next reminder times for a medication."""
    if not reminder.enabled:
        return []

    now = datetime.now()
    reminders = []

    for time_str in reminder.reminder_times:
        try:
            hour, minute = map(int, time_str.split(':'))
            reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # If today's reminder time has passed, schedule for tomorrow
            if reminder_time <= now:
                reminder_time += timedelta(days=1)

            reminders.append(reminder_time)
        except ValueError:
            continue

    return reminders


def generate_ai_chat_response(user_message: str, user_context: Optional[Dict] = None,
                              conversation_history: Optional[List] = None) -> str:
    """Generate AI chatbot response based on user message and context."""

    # Simple rule-based responses (in a real app, this would use a more sophisticated AI model)
    message_lower = user_message.lower()

    # Context-aware responses
    if user_context:
        current_mood = user_context.get("mood", "neutral")
        current_anxiety = user_context.get("anxiety_level", 0)

        if "how are you feeling" in message_lower or "how do you feel" in message_lower:
            if current_anxiety > 7:
                return "I notice you're experiencing high anxiety. Would you like to try some deep breathing exercises or talk about what's causing this stress?"
            elif current_mood < 0:
                return "I sense you're not feeling great today. Remember, it's okay to not be okay. Would you like to explore some mood-lifting activities?"
            else:
                return "I'm here to support you! How can I help you today?"

    # General responses
    if "medication" in message_lower or "medicine" in message_lower:
        return "I can help you track your medications and set reminders. Would you like to add a new medication reminder?"

    elif "migraine" in message_lower or "headache" in message_lower:
        return "I'm sorry you're experiencing a migraine. Have you tried resting in a dark room, staying hydrated, or taking your prescribed medication?"

    elif "anxiety" in message_lower or "stress" in message_lower:
        return "Anxiety can be really challenging. Would you like to try some grounding exercises or deep breathing techniques?"

    elif "sleep" in message_lower or "tired" in message_lower:
        return "Sleep is crucial for mental health. Are you having trouble falling asleep, staying asleep, or feeling rested?"

    elif "help" in message_lower:
        return "I'm here to help! I can assist with tracking your symptoms, setting medication reminders, analyzing patterns, or just listening. What do you need?"

    elif "thank" in message_lower:
        return "You're welcome! I'm here whenever you need support."

    else:
        return "I'm here to support your mental health journey. You can ask me about tracking symptoms, medications, patterns, or just chat about how you're feeling."


def export_data_to_pdf(entries: List[Entry], routines: List[Routine],
                       weather_data: List[WeatherData], user_profile: Optional[UserProfile] = None) -> str:
    """Export data to PDF format (returns file path)."""
    # This would use a library like reportlab to generate PDF
    # For now, return a placeholder
    return "data_export.pdf"


def export_data_to_json(entries: List[Entry], routines: List[Routine],
                        weather_data: List[WeatherData], user_profile: Optional[UserProfile] = None) -> dict:
    """Export data to JSON format."""
    export_data = {
        "export_date": datetime.now().isoformat(),
        "user_profile": user_profile_to_dict(user_profile) if user_profile else None,
        "entries": [entry_to_dict(entry) for entry in entries],
        "routines": [routine_to_dict(routine) for routine in routines],
        "weather_data": [weather_data_to_dict(weather) for weather in weather_data],
        "summary": {
            "total_entries": len(entries),
            "total_routines": len(routines),
            "date_range": {
                "start": min([e.date for e in entries]).isoformat() if entries else None,
                "end": max([e.date for e in entries]).isoformat() if entries else None
            }
        }
    }

    return export_data


# ---------------------------
# Wearables CSV import
# ---------------------------
def import_wearables_csv(db: Session, file: UploadFile) -> dict:
    """CSV columns: date (ISO), sleep_minutes (int), hrv (int)."""
    content = file.file.read()
    df = pd.read_csv(io.BytesIO(content))
    if "date" not in df.columns:
        raise ValueError("CSV must include 'date' column")

    has_sleep = "sleep_minutes" in df.columns
    has_hrv = "hrv" in df.columns

    updates = 0
    creates = 0
    for _, row in df.iterrows():
        try:
            d = datetime.fromisoformat(str(row["date"])).date()
        except Exception:
            d = datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
        e = db.query(Entry).filter(Entry.date == d).first()
        if not e:
            e = Entry(date=d)
            creates += 1
        if has_sleep and pd.notna(row.get("sleep_minutes")):
            try:
                e.sleep_hours = float(row["sleep_minutes"]) / 60.0
            except Exception:
                pass
        if has_hrv and pd.notna(row.get("hrv")):
            try:
                e.hrv = int(row["hrv"])
            except Exception:
                pass
        db.add(e)
        updates += 1
    db.commit()
    return {"updated_or_created": int(updates), "created_new": int(creates)}


# ---------------------------
# FastAPI app & routes
# ---------------------------
Base.metadata.create_all(bind=engine)
app = FastAPI(title="MindMap API (single-file)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "MindMap API (single-file)"}


@app.get("/profile", response_class=HTMLResponse, tags=["ui"])
def profile_interface():
    """User profile management interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MindTrack - Profile</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .section h3 { margin-top: 0; color: #333; }
            .form-group { margin: 15px 0; }
            .form-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
            .form-group input, .form-group select, .form-group textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
            .form-group textarea { height: 80px; resize: vertical; }
            .btn { padding: 12px 24px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px 5px; }
            .btn:hover { background: #45a049; }
            .btn-secondary { background: #2196F3; }
            .btn-secondary:hover { background: #1976D2; }
            .medication-item { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; background: #f9f9f9; }
            .medication-item input { margin: 5px 0; }
            .add-btn { background: #FF9800; }
            .add-btn:hover { background: #F57C00; }
            .remove-btn { background: #f44336; padding: 5px 10px; font-size: 12px; }
            .remove-btn:hover { background: #d32f2f; }
            .nav-links { text-align: center; margin: 20px 0; }
            .nav-links a { color: #2196F3; text-decoration: none; margin: 0 10px; }
            .nav-links a:hover { text-decoration: underline; }
            .success-message { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0; }
            .error-message { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MindTrack - Personal Profile</h1>
            <p>Manage your personal information, medical conditions, and current medications to get more personalized recommendations.</p>

            <div class="nav-links">
                <a href="/body-map">Body Map</a> | 
                <a href="/calendar">Calendar</a> | 
                <a href="/">Home</a>
            </div>

            <div id="messages"></div>

            <form id="profileForm">
                <div class="section">
                    <h3>Basic Information</h3>
                    <div class="form-group">
                        <label for="name">Name:</label>
                        <input type="text" id="name" name="name" placeholder="Your name">
                    </div>
                    <div class="form-group">
                        <label for="age">Age:</label>
                        <input type="number" id="age" name="age" min="0" max="120" placeholder="Your age">
                    </div>
                    <div class="form-group">
                        <label for="gender">Gender:</label>
                        <select id="gender" name="gender">
                            <option value="">Select gender</option>
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                            <option value="non-binary">Non-binary</option>
                            <option value="other">Other</option>
                        </select>
                    </div>
                </div>

                <div class="section">
                    <h3>Medical Information</h3>
                    <div class="form-group">
                        <label for="known_conditions">Known Conditions (comma-separated):</label>
                        <textarea id="known_conditions" name="known_conditions" placeholder="e.g., migraine, anxiety, depression, ADHD"></textarea>
                    </div>
                    <div class="form-group">
                        <label for="symptoms">Recurring Symptoms (comma-separated):</label>
                        <textarea id="symptoms" name="symptoms" placeholder="e.g., headaches, fatigue, insomnia"></textarea>
                    </div>
                    <div class="form-group">
                        <label for="allergies">Allergies (comma-separated):</label>
                        <textarea id="allergies" name="allergies" placeholder="e.g., penicillin, nuts, pollen"></textarea>
                    </div>
                </div>

                <div class="section">
                    <h3>Current Medications</h3>
                    <div id="medications-container">
                        <div class="medication-item">
                            <input type="text" name="med_name[]" placeholder="Medication name" required>
                            <input type="text" name="med_dose[]" placeholder="Dose (e.g., 10mg)">
                            <input type="text" name="med_frequency[]" placeholder="Frequency (e.g., daily, twice daily)">
                            <input type="text" name="med_notes[]" placeholder="Notes (optional)">
                        </div>
                    </div>
                    <button type="button" class="btn add-btn" onclick="addMedication()">Add Medication</button>
                </div>

                <div class="section">
                    <h3>Supplements</h3>
                    <div id="supplements-container">
                        <div class="medication-item">
                            <input type="text" name="supp_name[]" placeholder="Supplement name" required>
                            <input type="text" name="supp_dose[]" placeholder="Dose (e.g., 1000iu)">
                            <input type="text" name="supp_frequency[]" placeholder="Frequency (e.g., daily)">
                            <input type="text" name="supp_notes[]" placeholder="Notes (optional)">
                        </div>
                    </div>
                    <button type="button" class="btn add-btn" onclick="addSupplement()">Add Supplement</button>
                </div>

                <div class="section">
                    <h3>Preferences</h3>
                    <div class="form-group">
                        <label for="preferred_interventions">Preferred Interventions (comma-separated):</label>
                        <textarea id="preferred_interventions" name="preferred_interventions" placeholder="e.g., meditation, exercise, deep breathing"></textarea>
                    </div>
                    <div class="form-group">
                        <label for="emergency_contact">Emergency Contact:</label>
                        <input type="text" id="emergency_contact" name="emergency_contact" placeholder="Name and phone number">
                    </div>
                </div>

                <div style="text-align: center;">
                    <button type="submit" class="btn">Save Profile</button>
                    <button type="button" class="btn btn-secondary" onclick="loadProfile()">Load Profile</button>
                </div>
            </form>
        </div>

        <script>
            // Load profile on page load
            window.onload = function() {
                loadProfile();
            };

            // Add medication field
            function addMedication() {
                const container = document.getElementById('medications-container');
                const div = document.createElement('div');
                div.className = 'medication-item';
                div.innerHTML = `
                    <input type="text" name="med_name[]" placeholder="Medication name" required>
                    <input type="text" name="med_dose[]" placeholder="Dose (e.g., 10mg)">
                    <input type="text" name="med_frequency[]" placeholder="Frequency (e.g., daily, twice daily)">
                    <input type="text" name="med_notes[]" placeholder="Notes (optional)">
                    <button type="button" class="btn remove-btn" onclick="removeItem(this)">Remove</button>
                `;
                container.appendChild(div);
            }

            // Add supplement field
            function addSupplement() {
                const container = document.getElementById('supplements-container');
                const div = document.createElement('div');
                div.className = 'medication-item';
                div.innerHTML = `
                    <input type="text" name="supp_name[]" placeholder="Supplement name" required>
                    <input type="text" name="supp_dose[]" placeholder="Dose (e.g., 1000iu)">
                    <input type="text" name="supp_frequency[]" placeholder="Frequency (e.g., daily)">
                    <input type="text" name="supp_notes[]" placeholder="Notes (optional)">
                    <button type="button" class="btn remove-btn" onclick="removeItem(this)">Remove</button>
                `;
                container.appendChild(div);
            }

            // Remove item
            function removeItem(button) {
                button.parentElement.remove();
            }

            // Load profile
            async function loadProfile() {
                try {
                    const response = await fetch('/profile');
                    if (response.ok) {
                        const profile = await response.json();
                        populateForm(profile);
                        showMessage('Profile loaded successfully!', 'success');
                    } else if (response.status === 404) {
                        showMessage('No profile found. Fill out the form to create one.', 'info');
                    }
                } catch (error) {
                    showMessage('Error loading profile: ' + error.message, 'error');
                }
            }

            // Populate form with profile data
            function populateForm(profile) {
                document.getElementById('name').value = profile.name || '';
                document.getElementById('age').value = profile.age || '';
                document.getElementById('gender').value = profile.gender || '';
                document.getElementById('known_conditions').value = (profile.known_conditions || []).join(', ');
                document.getElementById('symptoms').value = (profile.symptoms || []).join(', ');
                document.getElementById('allergies').value = (profile.allergies || []).join(', ');
                document.getElementById('preferred_interventions').value = (profile.preferred_interventions || []).join(', ');
                document.getElementById('emergency_contact').value = profile.emergency_contact || '';

                // Populate medications
                populateMedications(profile.current_medications || [], 'medications-container', 'med');

                // Populate supplements
                populateMedications(profile.supplements || [], 'supplements-container', 'supp');
            }

            // Populate medications/supplements
            function populateMedications(items, containerId, prefix) {
                const container = document.getElementById(containerId);
                container.innerHTML = '';

                if (items.length === 0) {
                    // Add one empty item
                    if (prefix === 'med') {
                        addMedication();
                    } else {
                        addSupplement();
                    }
                    return;
                }

                items.forEach((item, index) => {
                    const div = document.createElement('div');
                    div.className = 'medication-item';
                    div.innerHTML = `
                        <input type="text" name="${prefix}_name[]" placeholder="${prefix === 'med' ? 'Medication' : 'Supplement'} name" value="${item.name || ''}" required>
                        <input type="text" name="${prefix}_dose[]" placeholder="Dose" value="${item.dose || ''}">
                        <input type="text" name="${prefix}_frequency[]" placeholder="Frequency" value="${item.frequency || ''}">
                        <input type="text" name="${prefix}_notes[]" placeholder="Notes" value="${item.notes || ''}">
                        ${index > 0 ? '<button type="button" class="btn remove-btn" onclick="removeItem(this)">Remove</button>' : ''}
                    `;
                    container.appendChild(div);
                });
            }

            // Handle form submission
            document.getElementById('profileForm').addEventListener('submit', async function(e) {
                e.preventDefault();

                const formData = new FormData(e.target);
                const profile = {
                    name: formData.get('name'),
                    age: formData.get('age') ? parseInt(formData.get('age')) : null,
                    gender: formData.get('gender'),
                    known_conditions: formData.get('known_conditions') ? formData.get('known_conditions').split(',').map(s => s.trim()).filter(s => s) : [],
                    symptoms: formData.get('symptoms') ? formData.get('symptoms').split(',').map(s => s.trim()).filter(s => s) : [],
                    allergies: formData.get('allergies') ? formData.get('allergies').split(',').map(s => s.trim()).filter(s => s) : [],
                    preferred_interventions: formData.get('preferred_interventions') ? formData.get('preferred_interventions').split(',').map(s => s.trim()).filter(s => s) : [],
                    emergency_contact: formData.get('emergency_contact'),
                    current_medications: collectMedications('med'),
                    supplements: collectMedications('supp')
                };

                try {
                    const response = await fetch('/profile', {
                        method: 'PUT',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(profile)
                    });

                    if (response.ok) {
                        showMessage('Profile saved successfully!', 'success');
                    } else {
                        const error = await response.json();
                        showMessage('Error saving profile: ' + error.detail, 'error');
                    }
                } catch (error) {
                    showMessage('Error saving profile: ' + error.message, 'error');
                }
            });

            // Collect medications/supplements from form
            function collectMedications(prefix) {
                const names = Array.from(document.getElementsByName(prefix + '_name[]')).map(el => el.value).filter(v => v);
                const doses = Array.from(document.getElementsByName(prefix + '_dose[]')).map(el => el.value);
                const frequencies = Array.from(document.getElementsByName(prefix + '_frequency[]')).map(el => el.value);
                const notes = Array.from(document.getElementsByName(prefix + '_notes[]')).map(el => el.value);

                return names.map((name, index) => ({
                    name: name,
                    dose: doses[index] || null,
                    frequency: frequencies[index] || null,
                    notes: notes[index] || null
                }));
            }

            // Show message
            function showMessage(message, type) {
                const messagesDiv = document.getElementById('messages');
                const div = document.createElement('div');
                div.className = type === 'success' ? 'success-message' : 'error-message';
                div.textContent = message;
                messagesDiv.appendChild(div);

                setTimeout(() => {
                    div.remove();
                }, 5000);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/routines", response_class=HTMLResponse, tags=["ui"])
def routine_tracking_interface():
    """Routine tracking interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MindTrack - Routine Tracking</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .nav-links { text-align: center; margin: 20px 0; }
            .nav-links a { color: #2196F3; text-decoration: none; margin: 0 10px; }
            .nav-links a:hover { text-decoration: underline; }

            .routine-form { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9; }
            .form-group { margin: 15px 0; }
            .form-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
            .form-group input, .form-group select, .form-group textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
            .form-group textarea { height: 80px; resize: vertical; }

            .btn { padding: 12px 24px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px 5px; }
            .btn:hover { background: #45a049; }
            .btn-secondary { background: #2196F3; }
            .btn-secondary:hover { background: #1976D2; }
            .btn-danger { background: #f44336; }
            .btn-danger:hover { background: #d32f2f; }

            .routines-list { margin: 20px 0; }
            .routine-item { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; background: white; }
            .routine-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
            .routine-type { font-weight: bold; color: #333; font-size: 18px; }
            .routine-time { color: #666; }
            .routine-details { margin: 10px 0; }
            .effectiveness-scores { display: flex; gap: 20px; margin: 10px 0; }
            .score-item { text-align: center; }
            .score-label { font-size: 12px; color: #666; }
            .score-value { font-size: 18px; font-weight: bold; color: #333; }

            .rating-slider { width: 100%; margin: 10px 0; }
            .rating-labels { display: flex; justify-content: space-between; font-size: 12px; color: #666; }

            .success-message { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0; }
            .error-message { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin: 10px 0; }

            .tabs { display: flex; margin: 20px 0; }
            .tab { padding: 10px 20px; background: #f0f0f0; border: none; cursor: pointer; }
            .tab.active { background: #4CAF50; color: white; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }

            .analytics-section { margin: 20px 0; padding: 20px; background: #e8f5e8; border-radius: 5px; }
            .recommendation-item { margin: 10px 0; padding: 10px; background: white; border-radius: 5px; border-left: 4px solid #4CAF50; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>MindTrack - Routine Tracking</h1>
                <p>Track your daily routines, activities, and their effectiveness in improving your symptoms, emotions, and sensations.</p>
            </div>

            <div class="nav-links">
                <a href="/body-map">Body Map</a> | 
                <a href="/profile">Profile</a> | 
                <a href="/calendar">Calendar</a> | 
                <a href="/">Home</a>
            </div>

            <div id="messages"></div>

            <div class="tabs">
                <button class="tab active" onclick="showTab('tracking')">Track Routines</button>
                <button class="tab" onclick="showTab('view')">View Today's Routines</button>
                <button class="tab" onclick="showTab('analytics')">Analytics</button>
            </div>

            <!-- Routine Tracking Tab -->
            <div id="tracking" class="tab-content active">
                <div class="routine-form">
                    <h3>Add New Routine</h3>
                    <form id="routineForm">
                        <div class="form-group">
                            <label for="routine_type">Routine Type:</label>
                            <select id="routine_type" name="routine_type" required>
                                <option value="">Select routine type</option>
                                <option value="morning">Morning Routine</option>
                                <option value="work">Work Routine</option>
                                <option value="school">School Routine</option>
                                <option value="exercise">Exercise Routine</option>
                                <option value="afternoon">Afternoon Routine</option>
                                <option value="evening">Evening Routine</option>
                            </select>
                        </div>

                        <div class="form-group">
                            <label for="start_time">Start Time:</label>
                            <input type="time" id="start_time" name="start_time">
                        </div>

                        <div class="form-group">
                            <label for="end_time">End Time:</label>
                            <input type="time" id="end_time" name="end_time">
                        </div>

                        <div class="form-group">
                            <label for="activities">Activities (comma-separated):</label>
                            <textarea id="activities" name="activities" placeholder="e.g., meditation, stretching, reading, walking"></textarea>
                        </div>

                        <div class="form-group">
                            <label for="notes">Notes:</label>
                            <textarea id="notes" name="notes" placeholder="Additional notes about this routine"></textarea>
                        </div>

                        <h4>Effectiveness Tracking</h4>

                        <div class="form-group">
                            <label for="symptoms_improved">Symptoms Improved (comma-separated):</label>
                            <textarea id="symptoms_improved" name="symptoms_improved" placeholder="e.g., headache, fatigue, anxiety"></textarea>
                        </div>

                        <div class="form-group">
                            <label for="emotions_improved">Emotions Improved (comma-separated):</label>
                            <textarea id="emotions_improved" name="emotions_improved" placeholder="e.g., stress, sadness, anger"></textarea>
                        </div>

                        <div class="form-group">
                            <label for="sensations_improved">Body Sensations Improved (comma-separated):</label>
                            <textarea id="sensations_improved" name="sensations_improved" placeholder="e.g., chest tightness, muscle tension, stomach pain"></textarea>
                        </div>

                        <h4>Effectiveness Ratings (0-10 scale)</h4>

                        <div class="form-group">
                            <label>Overall Effectiveness: <span id="overall_value">5</span></label>
                            <input type="range" min="0" max="10" value="5" class="rating-slider" id="overall_effectiveness" name="overall_effectiveness">
                            <div class="rating-labels">
                                <span>Not helpful</span>
                                <span>Very helpful</span>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Energy Level After: <span id="energy_value">5</span></label>
                            <input type="range" min="0" max="10" value="5" class="rating-slider" id="energy_level_after" name="energy_level_after">
                            <div class="rating-labels">
                                <span>Very low</span>
                                <span>Very high</span>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Mood Improvement: <span id="mood_value">5</span></label>
                            <input type="range" min="0" max="10" value="5" class="rating-slider" id="mood_improvement" name="mood_improvement">
                            <div class="rating-labels">
                                <span>No improvement</span>
                                <span>Major improvement</span>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Symptom Relief: <span id="symptom_value">5</span></label>
                            <input type="range" min="0" max="10" value="5" class="rating-slider" id="symptom_relief" name="symptom_relief">
                            <div class="rating-labels">
                                <span>No relief</span>
                                <span>Complete relief</span>
                            </div>
                        </div>

                        <div class="form-group">
                            <label for="completion_percentage">Completion Percentage:</label>
                            <input type="number" id="completion_percentage" name="completion_percentage" min="0" max="100" value="100">
                        </div>

                        <div style="text-align: center;">
                            <button type="submit" class="btn">Save Routine</button>
                            <button type="button" class="btn btn-secondary" onclick="loadTodayRoutines()">Load Today's Routines</button>
                        </div>
                    </form>
                </div>
            </div>

            <!-- View Routines Tab -->
            <div id="view" class="tab-content">
                <h3>Today's Routines</h3>
                <div id="routinesList" class="routines-list">
                    <p>Loading routines...</p>
                </div>
            </div>

            <!-- Analytics Tab -->
            <div id="analytics" class="tab-content">
                <h3>Routine Analytics</h3>
                <div class="analytics-section">
                    <button class="btn btn-secondary" onclick="loadAnalytics()">Load Analytics</button>
                    <div id="analyticsContent"></div>
                </div>
            </div>
        </div>

        <script>
            let currentEntryId = null;

            // Initialize
            window.onload = function() {
                setupSliders();
                createTodayEntry();
            };

            // Setup rating sliders
            function setupSliders() {
                const sliders = ['overall_effectiveness', 'energy_level_after', 'mood_improvement', 'symptom_relief'];
                sliders.forEach(sliderId => {
                    const slider = document.getElementById(sliderId);
                    const valueSpan = document.getElementById(sliderId.replace('_', '_value'));
                    slider.oninput = function() {
                        valueSpan.textContent = this.value;
                    };
                });
            }

            // Create today's entry if it doesn't exist
            async function createTodayEntry() {
                const today = new Date().toISOString().split('T')[0];
                try {
                    const response = await fetch('/entries', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            date: today
                        })
                    });

                    if (response.ok) {
                        const entry = await response.json();
                        currentEntryId = entry.id;
                        loadTodayRoutines();
                    }
                } catch (error) {
                    console.error('Error creating entry:', error);
                }
            }

            // Show tab content
            function showTab(tabName) {
                // Hide all tab contents
                const tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(content => content.classList.remove('active'));

                // Remove active class from all tabs
                const tabs = document.querySelectorAll('.tab');
                tabs.forEach(tab => tab.classList.remove('active'));

                // Show selected tab content
                document.getElementById(tabName).classList.add('active');

                // Add active class to clicked tab
                event.target.classList.add('active');

                // Load data for specific tabs
                if (tabName === 'view') {
                    loadTodayRoutines();
                } else if (tabName === 'analytics') {
                    loadAnalytics();
                }
            }

            // Handle form submission
            document.getElementById('routineForm').addEventListener('submit', async function(e) {
                e.preventDefault();

                if (!currentEntryId) {
                    showMessage('Please wait while we set up today\'s entry...', 'error');
                    return;
                }

                const formData = new FormData(e.target);
                const routine = {
                    entry_id: currentEntryId,
                    routine_type: formData.get('routine_type'),
                    start_time: formData.get('start_time'),
                    end_time: formData.get('end_time'),
                    activities: formData.get('activities') ? formData.get('activities').split(',').map(s => s.trim()).filter(s => s) : [],
                    notes: formData.get('notes'),
                    symptoms_improved: formData.get('symptoms_improved') ? formData.get('symptoms_improved').split(',').map(s => s.trim()).filter(s => s) : [],
                    emotions_improved: formData.get('emotions_improved') ? formData.get('emotions_improved').split(',').map(s => s.trim()).filter(s => s) : [],
                    sensations_improved: formData.get('sensations_improved') ? formData.get('sensations_improved').split(',').map(s => s.trim()).filter(s => s) : [],
                    overall_effectiveness: parseInt(formData.get('overall_effectiveness')),
                    energy_level_after: parseInt(formData.get('energy_level_after')),
                    mood_improvement: parseInt(formData.get('mood_improvement')),
                    symptom_relief: parseInt(formData.get('symptom_relief')),
                    completion_percentage: parseInt(formData.get('completion_percentage'))
                };

                try {
                    const response = await fetch('/routines', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(routine)
                    });

                    if (response.ok) {
                        showMessage('Routine saved successfully!', 'success');
                        e.target.reset();
                        loadTodayRoutines();
                    } else {
                        const error = await response.json();
                        showMessage('Error saving routine: ' + error.detail, 'error');
                    }
                } catch (error) {
                    showMessage('Error saving routine: ' + error.message, 'error');
                }
            });

            // Load today's routines
            async function loadTodayRoutines() {
                if (!currentEntryId) return;

                try {
                    const response = await fetch(`/routines/entry/${currentEntryId}`);
                    if (response.ok) {
                        const routines = await response.json();
                        displayRoutines(routines);
                    }
                } catch (error) {
                    console.error('Error loading routines:', error);
                }
            }

            // Display routines
            function displayRoutines(routines) {
                const container = document.getElementById('routinesList');

                if (routines.length === 0) {
                    container.innerHTML = '<p>No routines recorded for today yet.</p>';
                    return;
                }

                let html = '';
                routines.forEach(routine => {
                    html += `
                        <div class="routine-item">
                            <div class="routine-header">
                                <div>
                                    <div class="routine-type">${formatRoutineType(routine.routine_type)}</div>
                                    <div class="routine-time">${routine.start_time || ''} - ${routine.end_time || ''}</div>
                                </div>
                                <div>
                                    <button class="btn btn-danger" onclick="deleteRoutine(${routine.id})">Delete</button>
                                </div>
                            </div>

                            <div class="routine-details">
                                ${routine.activities.length > 0 ? `<p><strong>Activities:</strong> ${routine.activities.join(', ')}</p>` : ''}
                                ${routine.notes ? `<p><strong>Notes:</strong> ${routine.notes}</p>` : ''}
                                ${routine.symptoms_improved.length > 0 ? `<p><strong>Symptoms Improved:</strong> ${routine.symptoms_improved.join(', ')}</p>` : ''}
                                ${routine.emotions_improved.length > 0 ? `<p><strong>Emotions Improved:</strong> ${routine.emotions_improved.join(', ')}</p>` : ''}
                                ${routine.sensations_improved.length > 0 ? `<p><strong>Sensations Improved:</strong> ${routine.sensations_improved.join(', ')}</p>` : ''}
                            </div>

                            <div class="effectiveness-scores">
                                <div class="score-item">
                                    <div class="score-label">Overall</div>
                                    <div class="score-value">${routine.overall_effectiveness}/10</div>
                                </div>
                                <div class="score-item">
                                    <div class="score-label">Energy</div>
                                    <div class="score-value">${routine.energy_level_after}/10</div>
                                </div>
                                <div class="score-item">
                                    <div class="score-label">Mood</div>
                                    <div class="score-value">${routine.mood_improvement}/10</div>
                                </div>
                                <div class="score-item">
                                    <div class="score-label">Symptom Relief</div>
                                    <div class="score-value">${routine.symptom_relief}/10</div>
                                </div>
                                <div class="score-item">
                                    <div class="score-label">Completion</div>
                                    <div class="score-value">${routine.completion_percentage}%</div>
                                </div>
                            </div>
                        </div>
                    `;
                });

                container.innerHTML = html;
            }

            // Delete routine
            async function deleteRoutine(routineId) {
                if (!confirm('Are you sure you want to delete this routine?')) return;

                try {
                    const response = await fetch(`/routines/${routineId}`, {
                        method: 'DELETE'
                    });

                    if (response.ok) {
                        showMessage('Routine deleted successfully!', 'success');
                        loadTodayRoutines();
                    } else {
                        showMessage('Error deleting routine', 'error');
                    }
                } catch (error) {
                    showMessage('Error deleting routine: ' + error.message, 'error');
                }
            }

            // Load analytics
            async function loadAnalytics() {
                try {
                    const response = await fetch('/analytics/routines?days=30');
                    if (response.ok) {
                        const analytics = await response.json();
                        displayAnalytics(analytics);
                    }
                } catch (error) {
                    console.error('Error loading analytics:', error);
                }
            }

            // Display analytics
            function displayAnalytics(analytics) {
                const container = document.getElementById('analyticsContent');

                let html = `
                    <h4>Analysis Period: ${analytics.analysis_period}</h4>
                    <p>Total routines analyzed: ${analytics.total_routines_analyzed}</p>
                `;

                if (analytics.recommendations && analytics.recommendations.length > 0) {
                    html += '<h4>Recommendations</h4>';
                    analytics.recommendations.forEach(rec => {
                        html += `
                            <div class="recommendation-item">
                                <h5>${formatRoutineType(rec.routine_type)}</h5>
                                <p><strong>Reason:</strong> ${rec.reason}</p>
                                <ul>
                                    ${rec.suggestions.map(s => `<li>${s}</li>`).join('')}
                                </ul>
                            </div>
                        `;
                    });
                }

                if (analytics.suggested_improvements && analytics.suggested_improvements.length > 0) {
                    html += '<h4>Areas for Improvement</h4>';
                    analytics.suggested_improvements.forEach(imp => {
                        html += `
                            <div class="recommendation-item">
                                <h5>${formatRoutineType(imp.routine_type)} (Current Score: ${imp.current_score}/10)</h5>
                                <ul>
                                    ${imp.suggestions.map(s => `<li>${s}</li>`).join('')}
                                </ul>
                            </div>
                        `;
                    });
                }

                container.innerHTML = html;
            }

            // Format routine type for display
            function formatRoutineType(type) {
                return type.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
            }

            // Show message
            function showMessage(message, type) {
                const messagesDiv = document.getElementById('messages');
                const div = document.createElement('div');
                div.className = type === 'success' ? 'success-message' : 'error-message';
                div.textContent = message;
                messagesDiv.appendChild(div);

                setTimeout(() => {
                    div.remove();
                }, 5000);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/body-map", response_class=HTMLResponse, tags=["ui"])
def body_map_interface():
    """Interactive body map interface for tracking sensations."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MindTrack - Body Map</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .body-map { position: relative; width: 300px; height: 600px; margin: 20px auto; border: 2px solid #333; }
            .body-region { position: absolute; border-radius: 50%; cursor: pointer; border: 2px solid #333; }
            .body-region:hover { opacity: 0.8; }
            .intensity-0 { background-color: #f0f0f0; }
            .intensity-1 { background-color: #e3f2fd; }
            .intensity-2 { background-color: #bbdefb; }
            .intensity-3 { background-color: #90caf9; }
            .intensity-4 { background-color: #ffcc80; }
            .intensity-5 { background-color: #ffb74d; }
            .intensity-6 { background-color: #ff9800; }
            .intensity-7 { background-color: #ff7043; }
            .intensity-8 { background-color: #f44336; }
            .intensity-9 { background-color: #d32f2f; }
            .intensity-10 { background-color: #b71c1c; }
            .controls { text-align: center; margin: 20px 0; }
            .intensity-slider { width: 200px; margin: 10px; }
            .save-btn { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .save-btn:hover { background: #45a049; }
            .legend { margin: 20px 0; text-align: center; }
            .legend-item { display: inline-block; margin: 0 10px; }
            .legend-color { width: 20px; height: 20px; display: inline-block; border: 1px solid #333; }
            .recommendations { margin: 20px 0; padding: 20px; background: #f9f9f9; border-radius: 10px; }
            .recommendation-item { margin: 10px 0; padding: 10px; background: white; border-radius: 5px; border-left: 4px solid #4CAF50; }
            .urgency-high { border-left-color: #f44336; }
            .urgency-medium { border-left-color: #ff9800; }
            .urgency-low { border-left-color: #4CAF50; }
            .intervention-btn { margin: 5px; padding: 8px 15px; background: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .intervention-btn:hover { background: #1976D2; }
            .medication-suggestion { background: #fff3cd; border-left-color: #ffc107; }
            .recommendations-title { font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #333; }
            .body-part-section { margin: 15px 0; }
            .body-part-title { font-weight: bold; color: #666; }
            .overall-recommendations { background: #e8f5e8; border-left-color: #4CAF50; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MindTrack - Body Sensations Map</h1>
            <p>Click on body regions to track where you're feeling sensations or emotions. Use the slider to set intensity (0-10).</p>

            <div class="nav-links">
                <a href="/routines">Routines</a> | 
                <a href="/profile">Profile</a> | 
                <a href="/calendar">Calendar</a> | 
                <a href="/">Home</a>
            </div>

            <div class="controls">
                <label>Intensity: <span id="intensity-value">0</span></label><br>
                <input type="range" min="0" max="10" value="0" class="intensity-slider" id="intensity-slider">
                <br>
                <button class="save-btn" onclick="saveSensations()">Save Today's Sensations</button>
            </div>

            <div class="body-map" id="body-map">
                <!-- Body regions will be added here -->
            </div>

            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color intensity-0"></div> None (0)
                </div>
                <div class="legend-item">
                    <div class="legend-color intensity-3"></div> Low (1-3)
                </div>
                <div class="legend-item">
                    <div class="legend-color intensity-6"></div> Medium (4-6)
                </div>
                <div class="legend-item">
                    <div class="legend-color intensity-9"></div> High (7-10)
                </div>
            </div>

            <div class="recommendations" id="recommendations" style="display: none;">
                <div class="recommendations-title">Personalized Recommendations</div>
                <div id="recommendations-content"></div>
            </div>
        </div>

        <script>
            const bodyRegions = {
                'head': {x: 150, y: 60, radius: 40, label: 'Head'},
                'neck': {x: 150, y: 120, radius: 30, label: 'Neck'},
                'shoulders': {x: 150, y: 160, radius: 60, label: 'Shoulders'},
                'chest': {x: 150, y: 210, radius: 50, label: 'Chest'},
                'arms': {x: 90, y: 240, radius: 25, label: 'Arms'},
                'hands': {x: 75, y: 360, radius: 20, label: 'Hands'},
                'stomach': {x: 150, y: 300, radius: 40, label: 'Stomach'},
                'hips': {x: 150, y: 390, radius: 50, label: 'Hips'},
                'legs': {x: 150, y: 480, radius: 40, label: 'Legs'},
                'feet': {x: 150, y: 570, radius: 30, label: 'Feet'}
            };

            let sensations = {};
            let currentIntensity = 0;

            // Initialize body map
            function initBodyMap() {
                const bodyMap = document.getElementById('body-map');

                Object.entries(bodyRegions).forEach(([region, data]) => {
                    const div = document.createElement('div');
                    div.className = 'body-region intensity-0';
                    div.style.left = (data.x - data.radius) + 'px';
                    div.style.top = (data.y - data.radius) + 'px';
                    div.style.width = (data.radius * 2) + 'px';
                    div.style.height = (data.radius * 2) + 'px';
                    div.title = data.label + ' (Click to set intensity)';
                    div.onclick = () => setRegionIntensity(region, currentIntensity);
                    bodyMap.appendChild(div);
                });
            }

            // Set intensity for a region
            function setRegionIntensity(region, intensity) {
                if (intensity === 0) {
                    delete sensations[region];
                } else {
                    sensations[region] = intensity;
                }
                updateBodyMap();

                // Get recommendations if there are sensations
                if (Object.keys(sensations).length > 0) {
                    getRecommendations();
                } else {
                    document.getElementById('recommendations').style.display = 'none';
                }
            }

            // Update body map display
            function updateBodyMap() {
                const regions = document.querySelectorAll('.body-region');
                regions.forEach((region, index) => {
                    const regionName = Object.keys(bodyRegions)[index];
                    const intensity = sensations[regionName] || 0;
                    region.className = 'body-region intensity-' + intensity;
                });
            }

            // Get recommendations from API
            async function getRecommendations() {
                try {
                    const response = await fetch('/recommendations/interventions?days=30', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(sensations)
                    });

                    if (response.ok) {
                        const data = await response.json();
                        displayRecommendations(data.recommendations);
                    } else {
                        console.error('Error getting recommendations:', response.statusText);
                    }
                } catch (error) {
                    console.error('Error getting recommendations:', error.message);
                }
            }

            // Display recommendations
            function displayRecommendations(recommendations) {
                const container = document.getElementById('recommendations-content');
                const urgencyClass = 'urgency-' + recommendations.urgency_level;

                let html = '';

                // Overall recommendations
                if (recommendations.overall_recommendations && recommendations.overall_recommendations.length > 0) {
                    html += '<div class="recommendation-item overall-recommendations">';
                    html += '<div class="body-part-title">Top Recommendations</div>';
                    recommendations.overall_recommendations.forEach(intervention => {
                        html += `<button class="intervention-btn" onclick="logIntervention('${intervention}')">${formatInterventionName(intervention)}</button>`;
                    });
                    html += '</div>';
                }

                // Body part specific recommendations
                for (const [bodyPart, data] of Object.entries(recommendations.body_part_recommendations)) {
                    html += `<div class="recommendation-item urgency-${data.urgency}">`;
                    html += `<div class="body-part-title">${formatBodyPartName(bodyPart)} (Intensity: ${data.intensity})</div>`;
                    data.interventions.forEach(intervention => {
                        html += `<button class="intervention-btn" onclick="logIntervention('${intervention}', '${bodyPart}')">${formatInterventionName(intervention)}</button>`;
                    });
                    html += '</div>';
                }

                // Medication suggestions
                if (recommendations.medication_suggestions && recommendations.medication_suggestions.length > 0) {
                    recommendations.medication_suggestions.forEach(suggestion => {
                        html += '<div class="recommendation-item medication-suggestion">';
                        html += `<div class="body-part-title">${suggestion.reason}</div>`;
                        suggestion.suggestions.forEach(med => {
                            html += `<button class="intervention-btn" onclick="logIntervention('${med}', 'medication')">${formatInterventionName(med)}</button>`;
                        });
                        html += '</div>';
                    });
                }

                container.innerHTML = html;
                document.getElementById('recommendations').style.display = 'block';
            }

            // Format intervention names for display
            function formatInterventionName(intervention) {
                return intervention.replace(/_/g, ' ').split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
            }

            // Format body part names for display
            function formatBodyPartName(bodyPart) {
                return bodyPart.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
            }

            // Log when user clicks on an intervention
            function logIntervention(intervention, bodyPart = '') {
                console.log(`User selected intervention: ${intervention} for ${bodyPart}`);
                // Here you could track which interventions users actually try
                alert(`Great choice! Consider trying: ${formatInterventionName(intervention)}`);
            }

            // Save sensations to API
            async function saveSensations() {
                if (Object.keys(sensations).length === 0) {
                    alert('No sensations recorded. Click on body regions first.');
                    return;
                }

                const today = new Date().toISOString().split('T')[0];
                const payload = {
                    date: today,
                    body_sensations: sensations
                };

                try {
                    const response = await fetch('/entries', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(payload)
                    });

                    if (response.ok) {
                        alert('Sensations saved successfully!');
                        sensations = {};
                        updateBodyMap();
                        document.getElementById('recommendations').style.display = 'none';
                    } else {
                        alert('Error saving sensations: ' + response.statusText);
                    }
                } catch (error) {
                    alert('Error saving sensations: ' + error.message);
                }
            }

            // Initialize
            document.getElementById('intensity-slider').oninput = function() {
                currentIntensity = parseInt(this.value);
                document.getElementById('intensity-value').textContent = currentIntensity;
            };

            initBodyMap();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/calendar", response_class=HTMLResponse, tags=["ui"])
def calendar_interface():
    """Interactive calendar interface with streak tracking and retroactive logging."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MindTrack - Calendar & Streaks</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .streak-info { display: flex; justify-content: space-around; margin: 20px 0; }
            .streak-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; min-width: 150px; }
            .streak-number { font-size: 2em; font-weight: bold; }
            .streak-label { font-size: 0.9em; opacity: 0.9; }
            .calendar-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .calendar-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            .calendar-nav { display: flex; gap: 10px; }
            .nav-btn { padding: 10px 15px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .nav-btn:hover { background: #45a049; }
            .calendar-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 5px; }
            .calendar-day { padding: 10px; border: 1px solid #ddd; text-align: center; cursor: pointer; position: relative; }
            .calendar-day:hover { background: #f0f0f0; }
            .calendar-day.has-entry { background: #e8f5e8; border-color: #4CAF50; }
            .calendar-day.has-routines { background: #fff3cd; border-color: #ffc107; }
            .calendar-day.today { background: #e3f2fd; border-color: #2196F3; font-weight: bold; }
            .calendar-day.other-month { color: #ccc; }
            .day-number { font-weight: bold; }
            .day-indicators { position: absolute; top: 2px; right: 2px; font-size: 0.7em; }
            .indicator { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-left: 2px; }
            .indicator-entry { background: #4CAF50; }
            .indicator-routine { background: #ffc107; }
            .indicator-migraine { background: #f44336; }
            .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); }
            .modal-content { background-color: white; margin: 5% auto; padding: 20px; border-radius: 10px; width: 80%; max-width: 600px; max-height: 80vh; overflow-y: auto; }
            .close { color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer; }
            .close:hover { color: #000; }
            .form-group { margin: 15px 0; }
            .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
            .form-group input, .form-group select, .form-group textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            .form-row { display: flex; gap: 10px; }
            .form-row .form-group { flex: 1; }
            .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            .btn-primary { background: #4CAF50; color: white; }
            .btn-primary:hover { background: #45a049; }
            .btn-secondary { background: #2196F3; color: white; }
            .btn-secondary:hover { background: #1976D2; }
            .btn-danger { background: #f44336; color: white; }
            .btn-danger:hover { background: #d32f2f; }
            .suggestions { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 15px 0; }
            .suggestion-item { background: white; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #4CAF50; }
            .tabs { display: flex; margin-bottom: 20px; }
            .tab { padding: 10px 20px; background: #f0f0f0; border: none; cursor: pointer; }
            .tab.active { background: #4CAF50; color: white; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .nav-links { margin: 20px 0; }
            .nav-links a { margin-right: 15px; color: #4CAF50; text-decoration: none; }
            .nav-links a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>MindTrack - Calendar & Streaks</h1>
                <p>Track your daily entries, view streaks, and log past days with smart suggestions.</p>

                <div class="nav-links">
                    <a href="/routines">Routines</a> | 
                    <a href="/profile">Profile</a> | 
                    <a href="/body-map">Body Map</a> | 
                    <a href="/">Home</a>
                </div>
            </div>

            <div class="streak-info" id="streak-info">
                <!-- Streak information will be loaded here -->
            </div>

            <div class="tabs">
                <button class="tab active" onclick="showTab('calendar')">Calendar View</button>
                <button class="tab" onclick="showTab('retroactive')">Retroactive Log</button>
            </div>

            <div id="calendar-tab" class="tab-content active">
                <div class="calendar-container">
                    <div class="calendar-header">
                        <h2 id="current-month">Calendar</h2>
                        <div class="calendar-nav">
                            <button class="nav-btn" onclick="previousMonth()">← Previous</button>
                            <button class="nav-btn" onclick="nextMonth()">Next →</button>
                            <button class="nav-btn" onclick="goToToday()">Today</button>
                        </div>
                    </div>

                    <div class="calendar-grid" id="calendar-grid">
                        <!-- Calendar will be generated here -->
                    </div>
                </div>
            </div>

            <div id="retroactive-tab" class="tab-content">
                <div class="calendar-container">
                    <h2>Log Past Days</h2>
                    <p>Select a date to log an entry for a past day. We'll provide suggestions based on your patterns.</p>

                    <div class="form-group">
                        <label for="retroactive-date">Select Date:</label>
                        <input type="date" id="retroactive-date" onchange="loadRetroactiveSuggestions()">
                    </div>

                    <div id="retroactive-suggestions" class="suggestions" style="display: none;">
                        <!-- Suggestions will be loaded here -->
                    </div>

                    <button class="btn btn-primary" onclick="openRetroactiveForm()">Create Entry for Selected Date</button>
                </div>
            </div>
        </div>

        <!-- Day Details Modal -->
        <div id="day-modal" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeModal('day-modal')">&times;</span>
                <h2 id="modal-title">Day Details</h2>
                <div id="modal-content">
                    <!-- Day details will be loaded here -->
                </div>
            </div>
        </div>

        <!-- Retroactive Entry Modal -->
        <div id="retroactive-modal" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeModal('retroactive-modal')">&times;</span>
                <h2>Create Entry for <span id="retroactive-date-display"></span></h2>
                <form id="retroactive-form">
                    <div class="form-row">
                        <div class="form-group">
                            <label for="sleep-hours">Sleep Hours:</label>
                            <input type="number" id="sleep-hours" step="0.1" min="0" max="24">
                        </div>
                        <div class="form-group">
                            <label for="sleep-quality">Sleep Quality (1-5):</label>
                            <select id="sleep-quality">
                                <option value="">Select...</option>
                                <option value="1">1 - Poor</option>
                                <option value="2">2 - Fair</option>
                                <option value="3">3 - Good</option>
                                <option value="4">4 - Very Good</option>
                                <option value="5">5 - Excellent</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label for="mood">Mood (-3 to +3):</label>
                            <select id="mood">
                                <option value="">Select...</option>
                                <option value="-3">-3 - Very Negative</option>
                                <option value="-2">-2 - Negative</option>
                                <option value="-1">-1 - Slightly Negative</option>
                                <option value="0">0 - Neutral</option>
                                <option value="1">1 - Slightly Positive</option>
                                <option value="2">2 - Positive</option>
                                <option value="3">3 - Very Positive</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="anxiety">Anxiety Level (0-10):</label>
                            <input type="number" id="anxiety" min="0" max="10">
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label for="depression">Depression Level (0-10):</label>
                            <input type="number" id="depression" min="0" max="10">
                        </div>
                        <div class="form-group">
                            <label for="productivity">Productivity Score (0-100):</label>
                            <input type="number" id="productivity" min="0" max="100">
                        </div>
                    </div>

                    <div class="form-group">
                        <label for="migraine">Migraine:</label>
                        <select id="migraine">
                            <option value="false">No</option>
                            <option value="true">Yes</option>
                        </select>
                    </div>

                    <div class="form-group" id="migraine-intensity-group" style="display: none;">
                        <label for="migraine-intensity">Migraine Intensity (0-10):</label>
                        <input type="number" id="migraine-intensity" min="0" max="10">
                    </div>

                    <div class="form-group">
                        <label for="notes">Notes:</label>
                        <textarea id="notes" rows="3"></textarea>
                    </div>

                    <button type="submit" class="btn btn-primary">Create Entry</button>
                    <button type="button" class="btn btn-secondary" onclick="closeModal('retroactive-modal')">Cancel</button>
                </form>
            </div>
        </div>

        <script>
            let currentDate = new Date();
            let calendarData = {};
            let streakInfo = {};

            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                loadStreakInfo();
                loadCalendarData();
                setupEventListeners();
            });

            function setupEventListeners() {
                // Migraine toggle
                document.getElementById('migraine').addEventListener('change', function() {
                    const intensityGroup = document.getElementById('migraine-intensity-group');
                    intensityGroup.style.display = this.value === 'true' ? 'block' : 'none';
                });

                // Retroactive form submission
                document.getElementById('retroactive-form').addEventListener('submit', function(e) {
                    e.preventDefault();
                    submitRetroactiveEntry();
                });
            }

            function showTab(tabName) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));

                // Show selected tab
                document.getElementById(tabName + '-tab').classList.add('active');
                event.target.classList.add('active');
            }

            async function loadStreakInfo() {
                try {
                    const response = await fetch('/analytics/enhanced_streaks');
                    if (response.ok) {
                        streakInfo = await response.json();
                        displayStreakInfo();
                    }
                } catch (error) {
                    console.error('Error loading streak info:', error);
                }
            }

            function displayStreakInfo() {
                const container = document.getElementById('streak-info');
                const streakActive = streakInfo.streak_active ? 'active' : 'inactive';

                container.innerHTML = `
                    <div class="streak-card">
                        <div class="streak-number">${streakInfo.current_streak}</div>
                        <div class="streak-label">Current Streak</div>
                        <div class="streak-label">(${streakActive})</div>
                    </div>
                    <div class="streak-card">
                        <div class="streak-number">${streakInfo.best_streak}</div>
                        <div class="streak-label">Best Streak</div>
                    </div>
                    <div class="streak-card">
                        <div class="streak-number">${streakInfo.total_logged_days}</div>
                        <div class="streak-label">Total Days</div>
                    </div>
                    <div class="streak-card">
                        <div class="streak-number">${streakInfo.days_to_next_milestone || 'N/A'}</div>
                        <div class="streak-label">Days to Next Milestone</div>
                    </div>
                `;
            }

            async function loadCalendarData() {
                const start = new Date(currentDate.getFullYear(), currentDate.getMonth(), 1);
                const end = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 0);

                try {
                    const response = await fetch(`/analytics/calendar?start=${start.toISOString().split('T')[0]}&end=${end.toISOString().split('T')[0]}`);
                    if (response.ok) {
                        calendarData = await response.json();
                        renderCalendar();
                    }
                } catch (error) {
                    console.error('Error loading calendar data:', error);
                }
            }

            function renderCalendar() {
                const grid = document.getElementById('calendar-grid');
                const monthTitle = document.getElementById('current-month');

                // Set month title
                monthTitle.textContent = currentDate.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });

                // Clear grid
                grid.innerHTML = '';

                // Add day headers
                const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
                days.forEach(day => {
                    const dayHeader = document.createElement('div');
                    dayHeader.className = 'calendar-day';
                    dayHeader.style.fontWeight = 'bold';
                    dayHeader.style.backgroundColor = '#f0f0f0';
                    dayHeader.textContent = day;
                    grid.appendChild(dayHeader);
                });

                // Get first day of month and number of days
                const firstDay = new Date(currentDate.getFullYear(), currentDate.getMonth(), 1);
                const lastDay = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 0);
                const startDate = new Date(firstDay);
                startDate.setDate(startDate.getDate() - firstDay.getDay());

                // Generate calendar days
                for (let i = 0; i < 42; i++) {
                    const date = new Date(startDate);
                    date.setDate(startDate.getDate() + i);

                    const dayElement = document.createElement('div');
                    dayElement.className = 'calendar-day';

                    // Check if it's today
                    const today = new Date();
                    if (date.toDateString() === today.toDateString()) {
                        dayElement.classList.add('today');
                    }

                    // Check if it's in current month
                    if (date.getMonth() !== currentDate.getMonth()) {
                        dayElement.classList.add('other-month');
                    }

                    // Add day number
                    const dayNumber = document.createElement('div');
                    dayNumber.className = 'day-number';
                    dayNumber.textContent = date.getDate();
                    dayElement.appendChild(dayNumber);

                    // Add indicators
                    const indicators = document.createElement('div');
                    indicators.className = 'day-indicators';

                    const dateStr = date.toISOString().split('T')[0];
                    const dayData = calendarData.calendar_data.find(d => d.date === dateStr);

                    if (dayData) {
                        if (dayData.has_entry) {
                            dayElement.classList.add('has-entry');
                            const entryIndicator = document.createElement('div');
                            entryIndicator.className = 'indicator indicator-entry';
                            indicators.appendChild(entryIndicator);
                        }

                        if (dayData.has_routines) {
                            dayElement.classList.add('has-routines');
                            const routineIndicator = document.createElement('div');
                            routineIndicator.className = 'indicator indicator-routine';
                            indicators.appendChild(routineIndicator);
                        }

                        if (dayData.summary.migraine) {
                            const migraineIndicator = document.createElement('div');
                            migraineIndicator.className = 'indicator indicator-migraine';
                            indicators.appendChild(migraineIndicator);
                        }

                        // Add click handler
                        dayElement.onclick = () => showDayDetails(dateStr, dayData);
                    }

                    dayElement.appendChild(indicators);
                    grid.appendChild(dayElement);
                }
            }

            function showDayDetails(dateStr, dayData) {
                const modal = document.getElementById('day-modal');
                const title = document.getElementById('modal-title');
                const content = document.getElementById('modal-content');

                title.textContent = `Details for ${new Date(dateStr).toLocaleDateString()}`;

                let html = '';

                if (dayData.has_entry) {
                    const entry = dayData.entry_data;
                    html += '<h3>Daily Entry</h3>';
                    html += '<div style="background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0;">';
                    if (entry.sleep_hours) html += `<p><strong>Sleep:</strong> ${entry.sleep_hours} hours</p>`;
                    if (entry.mood !== null) html += `<p><strong>Mood:</strong> ${entry.mood}/3</p>`;
                    if (entry.anxiety !== null) html += `<p><strong>Anxiety:</strong> ${entry.anxiety}/10</p>`;
                    if (entry.migraine) html += `<p><strong>Migraine:</strong> Yes (Intensity: ${entry.migraine_intensity}/10)</p>`;
                    if (entry.body_sensations && Object.keys(entry.body_sensations).length > 0) {
                        html += '<p><strong>Body Sensations:</strong></p><ul>';
                        Object.entries(entry.body_sensations).forEach(([part, intensity]) => {
                            html += `<li>${part}: ${intensity}/10</li>`;
                        });
                        html += '</ul>';
                    }
                    html += '</div>';
                } else {
                    html += '<p>No entry recorded for this day.</p>';
                    html += `<button class="btn btn-primary" onclick="openRetroactiveForm('${dateStr}')">Log This Day</button>`;
                }

                if (dayData.has_routines) {
                    html += '<h3>Routines</h3>';
                    html += '<div style="background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0;">';
                    dayData.routines.forEach(routine => {
                        html += `<p><strong>${routine.type}:</strong> ${routine.duration} minutes (Effectiveness: ${routine.effectiveness}/10)</p>`;
                    });
                    html += '</div>';
                }

                content.innerHTML = html;
                modal.style.display = 'block';
            }

            async function loadRetroactiveSuggestions() {
                const selectedDate = document.getElementById('retroactive-date').value;
                if (!selectedDate) return;

                try {
                    const response = await fetch(`/analytics/retroactive_suggestions?target_date=${selectedDate}`);
                    if (response.ok) {
                        const data = await response.json();
                        displayRetroactiveSuggestions(data);
                    }
                } catch (error) {
                    console.error('Error loading suggestions:', error);
                }
            }

            function displayRetroactiveSuggestions(data) {
                const container = document.getElementById('retroactive-suggestions');

                if (data.suggestions.length === 0) {
                    container.innerHTML = '<p>No suggestions available for this date.</p>';
                } else {
                    let html = '<h3>Suggestions based on your patterns:</h3>';
                    data.suggestions.forEach(suggestion => {
                        html += `<div class="suggestion-item">${suggestion}</div>`;
                    });
                    container.innerHTML = html;
                }

                container.style.display = 'block';
            }

            function openRetroactiveForm(dateStr = null) {
                const modal = document.getElementById('retroactive-modal');
                const dateDisplay = document.getElementById('retroactive-date-display');

                if (dateStr) {
                    // Pre-fill the date input
                    document.getElementById('retroactive-date').value = dateStr;
                    dateDisplay.textContent = new Date(dateStr).toLocaleDateString();
                } else {
                    const selectedDate = document.getElementById('retroactive-date').value;
                    if (selectedDate) {
                        dateDisplay.textContent = new Date(selectedDate).toLocaleDateString();
                    } else {
                        dateDisplay.textContent = 'Selected Date';
                    }
                }

                modal.style.display = 'block';
            }

            async function submitRetroactiveEntry() {
                const selectedDate = document.getElementById('retroactive-date').value;
                if (!selectedDate) {
                    alert('Please select a date first.');
                    return;
                }

                const entry = {
                    date: selectedDate,
                    sleep_hours: parseFloat(document.getElementById('sleep-hours').value) || null,
                    sleep_quality: parseInt(document.getElementById('sleep-quality').value) || null,
                    mood_valence: parseInt(document.getElementById('mood').value) || null,
                    anxiety_level: parseInt(document.getElementById('anxiety').value) || null,
                    depression_level: parseInt(document.getElementById('depression').value) || null,
                    productivity_score: parseInt(document.getElementById('productivity').value) || null,
                    migraine: document.getElementById('migraine').value === 'true',
                    migraine_intensity: parseInt(document.getElementById('migraine-intensity').value) || null,
                    notes: document.getElementById('notes').value || null
                };

                try {
                    const response = await fetch('/entries/retroactive', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(entry)
                    });

                    if (response.ok) {
                        const result = await response.json();
                        alert('Entry created successfully!');
                        closeModal('retroactive-modal');
                        document.getElementById('retroactive-form').reset();
                        loadCalendarData();
                        loadStreakInfo();
                    } else {
                        const error = await response.json();
                        alert('Error creating entry: ' + error.detail);
                    }
                } catch (error) {
                    alert('Error creating entry: ' + error.message);
                }
            }

            function closeModal(modalId) {
                document.getElementById(modalId).style.display = 'none';
            }

            function previousMonth() {
                currentDate.setMonth(currentDate.getMonth() - 1);
                loadCalendarData();
            }

            function nextMonth() {
                currentDate.setMonth(currentDate.getMonth() + 1);
                loadCalendarData();
            }

            function goToToday() {
                currentDate = new Date();
                loadCalendarData();
            }

            // Close modal when clicking outside
            window.onclick = function(event) {
                if (event.target.classList.contains('modal')) {
                    event.target.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# --- Entries CRUD ---

@app.post("/entries", response_model=EntryOut, tags=["entries"])
def api_create_entry(payload: EntryCreate, db: Session = Depends(get_db)):
    obj = create_entry(db, payload)
    return entry_to_dict(obj)


@app.get("/entries", response_model=List[EntryOut], tags=["entries"])
def api_list_entries(
        start: Optional[date] = Query(default=None),
        end: Optional[date] = Query(default=None),
        limit: int = Query(default=5000, ge=1, le=100000),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(get_db),
):
    objs = get_entries(db, start=start, end=end, limit=limit, offset=offset)
    return [entry_to_dict(e) for e in objs]


@app.get("/entries/{entry_id}", response_model=EntryOut, tags=["entries"])
def api_get_entry(entry_id: int, db: Session = Depends(get_db)):
    obj = get_entry(db, entry_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry_to_dict(obj)


@app.put("/entries/{entry_id}", response_model=EntryOut, tags=["entries"])
def api_update_entry(entry_id: int, payload: EntryUpdate, db: Session = Depends(get_db)):
    obj = update_entry(db, entry_id, payload)
    if not obj:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry_to_dict(obj)


@app.delete("/entries/{entry_id}", tags=["entries"])
def api_delete_entry(entry_id: int, db: Session = Depends(get_db)):
    ok = delete_entry(db, entry_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"deleted": True, "id": entry_id}


@app.post("/entries/retroactive", response_model=EntryOut, tags=["entries"])
def api_create_retroactive_entry(
        payload: EntryCreate,
        db: Session = Depends(get_db)
):
    """Create an entry for a past date (retroactive logging)."""
    # Check if entry already exists for this date
    existing_entries = get_entries(db, start=payload.date, end=payload.date, limit=1)
    if existing_entries:
        raise HTTPException(
            status_code=400,
            detail=f"Entry already exists for {payload.date}. Use PUT to update existing entry."
        )

    # Create the entry
    obj = create_entry(db, payload)

    # Get retroactive suggestions for context
    entries = get_entries(db, limit=1000000)
    suggestions = get_retroactive_logging_suggestions(entries, payload.date)

    return {
        **entry_to_dict(obj),
        "retroactive_context": {
            "suggestions": suggestions["suggestions"],
            "patterns_used": suggestions["patterns"]
        }
    }


# --- Import/Export ---

@app.post("/import/wearables", tags=["import"])
def api_import_wearables(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        return import_wearables_csv(db, file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/export.csv", tags=["export"])
def api_export_csv(db: Session = Depends(get_db)):
    entries = get_entries(db, limit=1000000)
    records = [entry_to_dict(e) for e in entries]
    df = pd.DataFrame.from_records(records)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(content=buf.getvalue(), media_type="text/csv")


# --- Analytics ---

@app.get("/analytics/summary", tags=["analytics"])
def api_summary(
        days: int = Query(30, ge=1, le=365),
        db: Session = Depends(get_db)
):
    end = date.today()
    start = end - timedelta(days=days - 1)
    entries = get_entries(db, start=start, end=end, limit=1000000)
    return summary(entries)


@app.get("/analytics/trigger_risk", tags=["analytics"])
def api_trigger_risk(
        on_date: Optional[date] = Query(None, description="Compute risk for this date (default: today)"),
        db: Session = Depends(get_db)
):
    today = on_date or date.today()
    start = today - timedelta(days=1)
    entries = get_entries(db, start=start, end=today, limit=1000)
    return trigger_risk(entries, today)


@app.get("/analytics/streaks", tags=["analytics"])
def api_streaks(db: Session = Depends(get_db)):
    entries = get_entries(db, limit=1000000)
    return routines_scoring(entries)


@app.get("/analytics/calendar_heatmap", tags=["analytics"])
def api_calendar_heatmap(
        start: Optional[date] = Query(None),
        end: Optional[date] = Query(None),
        db: Session = Depends(get_db)
):
    if start is None or end is None:
        end = end or date.today()
        start = start or (end - timedelta(days=90))
    entries = get_entries(db, start=start, end=end, limit=1000000)
    return {"start": start.isoformat(), "end": end.isoformat(), "points": calendar_heatmap_points(entries, start, end)}


@app.get("/analytics/body_map", tags=["analytics"])
def api_body_map_analytics(
        days: int = Query(30, ge=1, le=365),
        db: Session = Depends(get_db)
):
    """Get body map analytics for the last N days."""
    end = date.today()
    start = end - timedelta(days=days - 1)
    entries = get_entries(db, start=start, end=end, limit=1000000)
    return body_map_analytics(entries)


@app.get("/analytics/body_map_heatmap", tags=["analytics"])
def api_body_map_heatmap(
        start: Optional[date] = Query(None),
        end: Optional[date] = Query(None),
        db: Session = Depends(get_db)
):
    """Get body map heatmap data for visualization."""
    if start is None or end is None:
        end = end or date.today()
        start = start or (end - timedelta(days=30))
    entries = get_entries(db, start=start, end=end, limit=1000000)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "heatmap_data": body_map_heatmap_data(entries, start, end)
    }


@app.post("/recommendations/interventions", tags=["recommendations"])
def api_get_intervention_recommendations(
        body_sensations: Dict[str, int],
        days: int = Query(30, ge=1, le=365, description="Days of historical data to analyze"),
        db: Session = Depends(get_db)
):
    """Get personalized intervention recommendations based on body sensations."""
    end = date.today()
    start = end - timedelta(days=days - 1)
    entries = get_entries(db, start=start, end=end, limit=1000000)
    user_profile = get_user_profile(db)

    recommendations = get_intervention_recommendations(body_sensations, entries, user_profile)
    return {
        "body_sensations": body_sensations,
        "analysis_period": f"{start} to {end}",
        "recommendations": recommendations
    }


# --- User Profile Management ---

@app.get("/profile", response_model=UserProfileOut, tags=["profile"])
def api_get_profile(db: Session = Depends(get_db)):
    """Get user profile."""
    profile = get_user_profile(db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return user_profile_to_dict(profile)


@app.post("/profile", response_model=UserProfileOut, tags=["profile"])
def api_create_profile(payload: UserProfileCreate, db: Session = Depends(get_db)):
    """Create user profile."""
    existing_profile = get_user_profile(db)
    if existing_profile:
        raise HTTPException(status_code=400, detail="Profile already exists. Use PUT to update.")

    profile = create_user_profile(db, payload)
    return user_profile_to_dict(profile)


@app.put("/profile", response_model=UserProfileOut, tags=["profile"])
def api_update_profile(payload: UserProfileUpdate, db: Session = Depends(get_db)):
    """Update user profile."""
    profile = update_user_profile(db, payload)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return user_profile_to_dict(profile)


# --- Routine Management ---

@app.post("/routines", response_model=RoutineOut, tags=["routines"])
def api_create_routine(payload: RoutineCreate, db: Session = Depends(get_db)):
    """Create a new routine entry."""
    # Verify that the entry exists
    entry = get_entry(db, payload.entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    routine = create_routine(db, payload)
    return routine_to_dict(routine)


@app.get("/routines/entry/{entry_id}", response_model=List[RoutineOut], tags=["routines"])
def api_get_routines_by_entry(entry_id: int, db: Session = Depends(get_db)):
    """Get all routines for a specific entry."""
    # Verify that the entry exists
    entry = get_entry(db, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    routines = get_routines_by_entry(db, entry_id)
    return [routine_to_dict(routine) for routine in routines]


@app.get("/routines/{routine_id}", response_model=RoutineOut, tags=["routines"])
def api_get_routine(routine_id: int, db: Session = Depends(get_db)):
    """Get a specific routine by ID."""
    routine = get_routine(db, routine_id)
    if not routine:
        raise HTTPException(status_code=404, detail="Routine not found")
    return routine_to_dict(routine)


@app.put("/routines/{routine_id}", response_model=RoutineOut, tags=["routines"])
def api_update_routine(routine_id: int, payload: RoutineUpdate, db: Session = Depends(get_db)):
    """Update a routine."""
    routine = update_routine(db, routine_id, payload)
    if not routine:
        raise HTTPException(status_code=404, detail="Routine not found")
    return routine_to_dict(routine)


@app.delete("/routines/{routine_id}", tags=["routines"])
def api_delete_routine(routine_id: int, db: Session = Depends(get_db)):
    """Delete a routine."""
    ok = delete_routine(db, routine_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Routine not found")
    return {"deleted": True, "id": routine_id}


@app.get("/analytics/routines", tags=["analytics"])
def api_routine_analytics(
        days: int = Query(30, ge=1, le=365, description="Days of data to analyze"),
        db: Session = Depends(get_db)
):
    """Get routine effectiveness analytics."""
    end = date.today()
    start = end - timedelta(days=days - 1)

    # Get entries in date range
    entries = get_entries(db, start=start, end=end, limit=1000000)
    entry_ids = [entry.id for entry in entries]

    # Get all routines for these entries
    all_routines = []
    for entry_id in entry_ids:
        routines = get_routines_by_entry(db, entry_id)
        all_routines.extend(routines)

    user_profile = get_user_profile(db)
    analysis = get_routine_recommendations(user_profile, all_routines)

    return {
        "analysis_period": f"{start} to {end}",
        "total_routines_analyzed": len(all_routines),
        **analysis
    }


@app.get("/analytics/calendar", tags=["analytics"])
def api_calendar_data(
        start: Optional[date] = Query(None),
        end: Optional[date] = Query(None),
        db: Session = Depends(get_db)
):
    """Get comprehensive calendar data including entries, routines, and streaks."""
    if start is None or end is None:
        end = end or date.today()
        start = start or (end - timedelta(days=30))

    entries = get_entries(db, start=start, end=end, limit=1000000)

    # Get all routines for the entries
    all_routines = []
    for entry in entries:
        routines = get_routines_by_entry(db, entry.id)
        all_routines.extend(routines)

    calendar_data = get_calendar_data(entries, all_routines, start, end)
    return calendar_data


@app.get("/analytics/enhanced_streaks", tags=["analytics"])
def api_enhanced_streaks(db: Session = Depends(get_db)):
    """Get enhanced streak information with milestones and progress."""
    entries = get_entries(db, limit=1000000)
    streak_info = get_logging_streak(entries)
    return streak_info


@app.get("/analytics/retroactive_suggestions", tags=["analytics"])
def api_retroactive_suggestions(
        target_date: date = Query(..., description="Date to get suggestions for"),
        db: Session = Depends(get_db)
):
    """Get suggestions for retroactive logging based on patterns."""
    entries = get_entries(db, limit=1000000)
    suggestions = get_retroactive_logging_suggestions(entries, target_date)
    return {
        "target_date": target_date.isoformat(),
        **suggestions
    }


# --- Charts (PNG) ---

@app.get("/charts/migraine_heatmap.png", tags=["charts"])
def api_migraine_heatmap_png(
        start: Optional[date] = Query(None),
        end: Optional[date] = Query(None),
        db: Session = Depends(get_db)
):
    if start is None or end is None:
        end = end or date.today()
        start = start or (end - timedelta(days=90))

    entries = get_entries(db, start=start, end=end, limit=1000000)
    data = {e.date: (e.migraine_intensity or 0) for e in entries}

    # Build weekly grid (rows = weeks, cols = weekdays Mon..Sun)
    days = (end - start).days + 1
    xs, ys, vs = [], [], []
    week_index = 0
    for i in range(days):
        d = start + timedelta(days=i)
        weekday = d.weekday()  # Mon=0..Sun=6
        if i == 0:
            week_index = 0
        else:
            if weekday == 0:
                week_index += 1
        xs.append(weekday)
        ys.append(week_index)
        vs.append(data.get(d, 0))

    height = (max(ys) + 1) if ys else 1
    width = 7
    grid = np.zeros((height, width))
    for x, y, v in zip(xs, ys, vs):
        grid[y, x] = v

    # IMPORTANT: no custom styles/colors per instructions
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(grid, aspect="auto")
    ax.set_title("Migraine Intensity Heatmap")
    ax.set_xlabel("Weekday (Mon=0 … Sun=6)")
    ax.set_ylabel("Week Index")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/charts/body_map.png", tags=["charts"])
def api_body_map_png(
        start: Optional[date] = Query(None),
        end: Optional[date] = Query(None),
        db: Session = Depends(get_db)
):
    """Generate body map heatmap as PNG image."""
    if start is None or end is None:
        end = end or date.today()
        start = start or (end - timedelta(days=30))

    entries = get_entries(db, start=start, end=end, limit=1000000)
    heatmap_data = body_map_heatmap_data(entries, start, end)

    # Create body map visualization
    fig, ax = plt.subplots(figsize=(8, 12))

    # Draw body outline (simplified)
    body_outline = plt.Rectangle((0.2, 0.05), 0.6, 0.9,
                                 fill=False, color='black', linewidth=2)
    ax.add_patch(body_outline)

    # Draw head
    head = plt.Circle((0.5, 0.1), 0.08, fill=False, color='black', linewidth=2)
    ax.add_patch(head)

    # Draw body regions with heatmap colors
    for region, data in heatmap_data.items():
        if data["intensity"] > 0:
            # Color intensity based on sensation level (0-10)
            intensity = data["intensity"]
            if intensity <= 3:
                color = 'lightblue'
            elif intensity <= 6:
                color = 'orange'
            else:
                color = 'red'

            # Draw region circle
            circle = plt.Circle((data["x"], data["y"]), data["radius"],
                                color=color, alpha=0.7)
            ax.add_patch(circle)

            # Add text label
            ax.text(data["x"], data["y"], f"{region}\n{intensity}",
                    ha='center', va='center', fontsize=8, weight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f"Body Sensations Heatmap\n{start} to {end}")
    ax.axis('off')

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.7, label='Low (1-3)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.7, label='Medium (4-6)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='High (7-10)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------
# Medication Reminder Endpoints
# ---------------------------

@app.post("/medication-reminders", response_model=MedicationReminderOut, tags=["medication-reminders"])
def api_create_medication_reminder(payload: MedicationReminderCreate, db: Session = Depends(get_db)):
    """Create a new medication reminder."""
    reminder = create_medication_reminder(db, payload)
    return medication_reminder_to_dict(reminder)


@app.get("/medication-reminders", response_model=List[MedicationReminderOut], tags=["medication-reminders"])
def api_get_medication_reminders(
        enabled_only: bool = Query(False, description="Only return enabled reminders"),
        db: Session = Depends(get_db)
):
    """Get all medication reminders."""
    reminders = get_medication_reminders(db, enabled_only=enabled_only)
    return [medication_reminder_to_dict(reminder) for reminder in reminders]


@app.get("/medication-reminders/{reminder_id}", response_model=MedicationReminderOut, tags=["medication-reminders"])
def api_get_medication_reminder(reminder_id: int, db: Session = Depends(get_db)):
    """Get a specific medication reminder."""
    reminder = get_medication_reminder(db, reminder_id)
    if not reminder:
        raise HTTPException(status_code=404, detail="Medication reminder not found")
    return medication_reminder_to_dict(reminder)


@app.put("/medication-reminders/{reminder_id}", response_model=MedicationReminderOut, tags=["medication-reminders"])
def api_update_medication_reminder(reminder_id: int, payload: MedicationReminderUpdate, db: Session = Depends(get_db)):
    """Update a medication reminder."""
    reminder = update_medication_reminder(db, reminder_id, payload)
    if not reminder:
        raise HTTPException(status_code=404, detail="Medication reminder not found")
    return medication_reminder_to_dict(reminder)


@app.delete("/medication-reminders/{reminder_id}", tags=["medication-reminders"])
def api_delete_medication_reminder(reminder_id: int, db: Session = Depends(get_db)):
    """Delete a medication reminder."""
    ok = delete_medication_reminder(db, reminder_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Medication reminder not found")
    return {"deleted": True, "id": reminder_id}


@app.post("/medication-reminders/{reminder_id}/mark-taken", tags=["medication-reminders"])
def api_mark_medication_taken(reminder_id: int, db: Session = Depends(get_db)):
    """Mark a medication as taken."""
    reminder = get_medication_reminder(db, reminder_id)
    if not reminder:
        raise HTTPException(status_code=404, detail="Medication reminder not found")

    reminder.last_taken = datetime.now()
    db.add(reminder)
    db.commit()
    db.refresh(reminder)

    return {"message": "Medication marked as taken", "reminder": medication_reminder_to_dict(reminder)}


# ---------------------------
# Weather Integration Endpoints
# ---------------------------

@app.post("/weather/fetch", tags=["weather"])
def api_fetch_weather_data(
        location: str = Query(..., description="City name or coordinates"),
        days: int = Query(7, ge=1, le=14, description="Number of days to fetch"),
        db: Session = Depends(get_db)
):
    """Fetch weather data from external API and store it."""
    try:
        # This would use a real weather API (OpenWeatherMap, WeatherAPI, etc.)
        # For demo purposes, we'll create mock data
        weather_data = []
        for i in range(days):
            mock_weather = WeatherDataCreate(
                date=date.today() + timedelta(days=i),
                temperature_high=75.0 + (i * 2),
                temperature_low=60.0 + (i * 1),
                humidity=65.0 + (i * 3),
                pressure=29.92 + (i * 0.01),
                wind_speed=8.0 + (i * 0.5),
                precipitation=0.0 if i % 3 != 0 else 0.1,
                weather_condition="sunny" if i % 2 == 0 else "cloudy",
                location=location
            )
            weather_obj = create_weather_data(db, mock_weather)
            weather_data.append(weather_data_to_dict(weather_obj))

        return {
            "message": f"Weather data fetched for {location}",
            "days_fetched": days,
            "weather_data": weather_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching weather data: {str(e)}")


@app.get("/weather/correlation", tags=["weather"])
def api_weather_correlation(
        days: int = Query(30, ge=1, le=365, description="Days of data to analyze"),
        db: Session = Depends(get_db)
):
    """Analyze correlation between weather and symptoms."""
    end = date.today()
    start = end - timedelta(days=days - 1)

    entries = get_entries(db, start=start, end=end, limit=1000000)
    weather_data = get_weather_data(db, start=start, end=end)

    correlation_analysis = analyze_weather_correlation(entries, weather_data)

    return {
        "analysis_period": f"{start} to {end}",
        "entries_analyzed": len(entries),
        "weather_days": len(weather_data),
        **correlation_analysis
    }


# ---------------------------
# Emergency Alert Endpoints
# ---------------------------

@app.post("/emergency-alerts", response_model=EmergencyAlertOut, tags=["emergency-alerts"])
def api_create_emergency_alert(payload: EmergencyAlertCreate, db: Session = Depends(get_db)):
    """Create a new emergency alert."""
    alert = create_emergency_alert(db, payload)
    return emergency_alert_to_dict(alert)


@app.get("/emergency-alerts", response_model=List[EmergencyAlertOut], tags=["emergency-alerts"])
def api_get_emergency_alerts(
        status: Optional[str] = Query(None, description="Filter by status: active, acknowledged, resolved"),
        db: Session = Depends(get_db)
):
    """Get emergency alerts."""
    alerts = get_emergency_alerts(db, status=status)
    return [emergency_alert_to_dict(alert) for alert in alerts]


@app.put("/emergency-alerts/{alert_id}/acknowledge", tags=["emergency-alerts"])
def api_acknowledge_emergency_alert(alert_id: int, db: Session = Depends(get_db)):
    """Acknowledge an emergency alert."""
    alert = db.get(EmergencyAlert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Emergency alert not found")

    alert.status = "acknowledged"
    alert.acknowledged_at = datetime.now()
    db.add(alert)
    db.commit()
    db.refresh(alert)

    return {"message": "Alert acknowledged", "alert": emergency_alert_to_dict(alert)}


@app.put("/emergency-alerts/{alert_id}/resolve", tags=["emergency-alerts"])
def api_resolve_emergency_alert(
        alert_id: int,
        response_action: str = Query(..., description="What action was taken"),
        db: Session = Depends(get_db)
):
    """Resolve an emergency alert."""
    alert = db.get(EmergencyAlert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Emergency alert not found")

    alert.status = "resolved"
    alert.resolved_at = datetime.now()
    alert.response_action = response_action
    db.add(alert)
    db.commit()
    db.refresh(alert)

    return {"message": "Alert resolved", "alert": emergency_alert_to_dict(alert)}


# ---------------------------
# AI Chatbot Endpoints
# ---------------------------

@app.websocket("/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str, db: Session = Depends(get_db)):
    """WebSocket endpoint for AI chatbot."""
    await websocket.accept()

    try:
        # Get or create chat session
        session = get_chat_session(db, session_id)
        if not session:
            session = create_chat_session(db, session_id)

        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")

            # Generate AI response
            ai_response = generate_ai_chat_response(
                user_message,
                session.user_context,
                session.conversation_history
            )

            # Update conversation history
            if not session.conversation_history:
                session.conversation_history = []

            session.conversation_history.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })

            session.conversation_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().isoformat()
            })

            # Keep only last 20 messages to prevent memory issues
            if len(session.conversation_history) > 20:
                session.conversation_history = session.conversation_history[-20:]

            # Update session in database
            update_chat_session(db, session_id,
                                conversation_history=session.conversation_history)

            # Send response back to client
            await websocket.send_text(json.dumps({
                "response": ai_response,
                "timestamp": datetime.now().isoformat()
            }))

    except WebSocketDisconnect:
        print(f"Client {session_id} disconnected")
    except Exception as e:
        await websocket.send_text(json.dumps({
            "error": f"An error occurred: {str(e)}"
        }))


@app.post("/chat/update-context", tags=["chat"])
def api_update_chat_context(
        session_id: str,
        context: Dict[str, Any],
        db: Session = Depends(get_db)
):
    """Update user context for AI chatbot."""
    session = get_chat_session(db, session_id)
    if not session:
        session = create_chat_session(db, session_id)

    updated_session = update_chat_session(db, session_id, user_context=context)
    return {"message": "Context updated", "session": chat_session_to_dict(updated_session)}


# ---------------------------
# Data Export Endpoints
# ---------------------------

@app.get("/export/json", tags=["export"])
def api_export_json(
        start: Optional[date] = Query(None),
        end: Optional[date] = Query(None),
        db: Session = Depends(get_db)
):
    """Export data to JSON format."""
    if start is None or end is None:
        end = end or date.today()
        start = start or (end - timedelta(days=30))

    entries = get_entries(db, start=start, end=end, limit=1000000)

    # Get all routines for the entries
    all_routines = []
    for entry in entries:
        routines = get_routines_by_entry(db, entry.id)
        all_routines.extend(routines)

    weather_data = get_weather_data(db, start=start, end=end)
    user_profile = get_user_profile(db)

    export_data = export_data_to_json(entries, all_routines, weather_data, user_profile)

    return export_data


@app.get("/export/csv", tags=["export"])
def api_export_csv(
        start: Optional[date] = Query(None),
        end: Optional[date] = Query(None),
        db: Session = Depends(get_db)
):
    """Export data to CSV format."""
    if start is None or end is None:
        end = end or date.today()
        start = start or (end - timedelta(days=30))

    entries = get_entries(db, start=start, end=end, limit=1000000)
    records = [entry_to_dict(e) for e in entries]
    df = pd.DataFrame.from_records(records)

    buf = io.StringIO()
    df.to_csv(buf, index=False)

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=mindtrack_export_{start}_{end}.csv"}
    )


@app.get("/export/pdf", tags=["export"])
def api_export_pdf(
        start: Optional[date] = Query(None),
        end: Optional[date] = Query(None),
        db: Session = Depends(get_db)
):
    """Export data to PDF format."""
    if start is None or end is None:
        end = end or date.today()
        start = start or (end - timedelta(days=30))

    entries = get_entries(db, start=start, end=end, limit=1000000)

    # Get all routines for the entries
    all_routines = []
    for entry in entries:
        routines = get_routines_by_entry(db, entry.id)
        all_routines.extend(routines)

    weather_data = get_weather_data(db, start=start, end=end)
    user_profile = get_user_profile(db)

    # For now, return a placeholder response
    # In a real implementation, this would generate an actual PDF
    return {
        "message": "PDF export functionality would be implemented here",
        "export_period": f"{start} to {end}",
        "entries_count": len(entries),
        "routines_count": len(all_routines),
        "weather_days": len(weather_data)
    }


# ---------------------------
# HTML Interfaces for Advanced Features
# ---------------------------

@app.get("/medication-reminders", response_class=HTMLResponse, tags=["ui"])
def medication_reminders_interface():
    """Medication reminders management interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MindTrack - Medication Reminders</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .nav-links { text-align: center; margin: 20px 0; }
            .nav-links a { color: #2196F3; text-decoration: none; margin: 0 10px; }
            .nav-links a:hover { text-decoration: underline; }

            .reminder-form { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9; }
            .form-group { margin: 15px 0; }
            .form-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
            .form-group input, .form-group select, .form-group textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
            .form-group textarea { height: 80px; resize: vertical; }

            .btn { padding: 12px 24px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px 5px; }
            .btn:hover { background: #45a049; }
            .btn-secondary { background: #2196F3; }
            .btn-secondary:hover { background: #1976D2; }
            .btn-danger { background: #f44336; }
            .btn-danger:hover { background: #d32f2f; }
            .btn-warning { background: #ff9800; }
            .btn-warning:hover { background: #F57C00; }

            .reminders-list { margin: 20px 0; }
            .reminder-item { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; background: white; }
            .reminder-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
            .reminder-name { font-weight: bold; color: #333; font-size: 18px; }
            .reminder-status { padding: 5px 10px; border-radius: 3px; font-size: 12px; }
            .status-enabled { background: #d4edda; color: #155724; }
            .status-disabled { background: #f8d7da; color: #721c24; }

            .reminder-details { margin: 10px 0; }
            .reminder-times { display: flex; gap: 10px; margin: 10px 0; }
            .time-badge { background: #e3f2fd; color: #1976d2; padding: 5px 10px; border-radius: 15px; font-size: 12px; }

            .success-message { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0; }
            .error-message { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin: 10px 0; }

            .time-inputs { display: flex; gap: 10px; align-items: center; }
            .time-inputs input { width: 100px; }
            .add-time-btn { background: #FF9800; padding: 8px 15px; }
            .add-time-btn:hover { background: #F57C00; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>MindTrack - Medication Reminders</h1>
                <p>Set up and manage medication reminders to help you stay on track with your treatment plan.</p>
            </div>

            <div class="nav-links">
                <a href="/routines">Routines</a> | 
                <a href="/profile">Profile</a> | 
                <a href="/body-map">Body Map</a> | 
                <a href="/calendar">Calendar</a> | 
                <a href="/">Home</a>
            </div>

            <div id="messages"></div>

            <div class="reminder-form">
                <h3>Add New Medication Reminder</h3>
                <form id="reminderForm">
                    <div class="form-group">
                        <label for="medication_name">Medication Name:</label>
                        <input type="text" id="medication_name" name="medication_name" required placeholder="e.g., Ibuprofen, Vitamin D">
                    </div>

                    <div class="form-group">
                        <label for="dose">Dose:</label>
                        <input type="text" id="dose" name="dose" placeholder="e.g., 200mg, 1000iu">
                    </div>

                    <div class="form-group">
                        <label for="frequency">Frequency:</label>
                        <select id="frequency" name="frequency" required>
                            <option value="">Select frequency</option>
                            <option value="daily">Daily</option>
                            <option value="twice_daily">Twice Daily</option>
                            <option value="three_times_daily">Three Times Daily</option>
                            <option value="weekly">Weekly</option>
                            <option value="custom">Custom</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Reminder Times:</label>
                        <div id="timeInputs" class="time-inputs">
                            <input type="time" id="time1" required>
                            <button type="button" class="btn add-time-btn" onclick="addTimeInput()">Add Time</button>
                        </div>
                    </div>

                    <div class="form-group">
                        <label for="notification_method">Notification Method:</label>
                        <select id="notification_method" name="notification_method">
                            <option value="app">App Notification</option>
                            <option value="email">Email</option>
                            <option value="sms">SMS</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="notes">Notes:</label>
                        <textarea id="notes" name="notes" placeholder="Additional notes about this medication"></textarea>
                    </div>

                    <div style="text-align: center;">
                        <button type="submit" class="btn">Create Reminder</button>
                        <button type="button" class="btn btn-secondary" onclick="loadReminders()">Load Reminders</button>
                    </div>
                </form>
            </div>

            <div class="reminders-list">
                <h3>Your Medication Reminders</h3>
                <div id="remindersList">
                    <p>Loading reminders...</p>
                </div>
            </div>
        </div>

        <script>
            let timeInputCount = 1;

            // Initialize
            window.onload = function() {
                loadReminders();
            };

            // Add time input
            function addTimeInput() {
                timeInputCount++;
                const container = document.getElementById('timeInputs');
                const newInput = document.createElement('input');
                newInput.type = 'time';
                newInput.id = 'time' + timeInputCount;
                newInput.required = true;
                container.insertBefore(newInput, container.lastElementChild);
            }

            // Handle form submission
            document.getElementById('reminderForm').addEventListener('submit', async function(e) {
                e.preventDefault();

                const formData = new FormData(e.target);
                const times = [];
                for (let i = 1; i <= timeInputCount; i++) {
                    const timeInput = document.getElementById('time' + i);
                    if (timeInput && timeInput.value) {
                        times.push(timeInput.value);
                    }
                }

                const reminder = {
                    medication_name: formData.get('medication_name'),
                    dose: formData.get('dose'),
                    frequency: formData.get('frequency'),
                    reminder_times: times,
                    notification_method: formData.get('notification_method'),
                    notes: formData.get('notes'),
                    enabled: true
                };

                try {
                    const response = await fetch('/medication-reminders', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(reminder)
                    });

                    if (response.ok) {
                        showMessage('Reminder created successfully!', 'success');
                        e.target.reset();
                        timeInputCount = 1;
                        document.getElementById('timeInputs').innerHTML = `
                            <input type="time" id="time1" required>
                            <button type="button" class="btn add-time-btn" onclick="addTimeInput()">Add Time</button>
                        `;
                        loadReminders();
                    } else {
                        const error = await response.json();
                        showMessage('Error creating reminder: ' + error.detail, 'error');
                    }
                } catch (error) {
                    showMessage('Error creating reminder: ' + error.message, 'error');
                }
            });

            // Load reminders
            async function loadReminders() {
                try {
                    const response = await fetch('/medication-reminders');
                    if (response.ok) {
                        const reminders = await response.json();
                        displayReminders(reminders);
                    }
                } catch (error) {
                    console.error('Error loading reminders:', error);
                }
            }

            // Display reminders
            function displayReminders(reminders) {
                const container = document.getElementById('remindersList');

                if (reminders.length === 0) {
                    container.innerHTML = '<p>No medication reminders set up yet.</p>';
                    return;
                }

                let html = '';
                reminders.forEach(reminder => {
                    const statusClass = reminder.enabled ? 'status-enabled' : 'status-disabled';
                    const statusText = reminder.enabled ? 'Enabled' : 'Disabled';

                    html += `
                        <div class="reminder-item">
                            <div class="reminder-header">
                                <div>
                                    <div class="reminder-name">${reminder.medication_name}</div>
                                    <span class="reminder-status ${statusClass}">${statusText}</span>
                                </div>
                                <div>
                                    <button class="btn btn-warning" onclick="markTaken(${reminder.id})">Mark Taken</button>
                                    <button class="btn btn-secondary" onclick="toggleReminder(${reminder.id}, ${!reminder.enabled})">${reminder.enabled ? 'Disable' : 'Enable'}</button>
                                    <button class="btn btn-danger" onclick="deleteReminder(${reminder.id})">Delete</button>
                                </div>
                            </div>

                            <div class="reminder-details">
                                ${reminder.dose ? `<p><strong>Dose:</strong> ${reminder.dose}</p>` : ''}
                                <p><strong>Frequency:</strong> ${formatFrequency(reminder.frequency)}</p>
                                <p><strong>Notification:</strong> ${reminder.notification_method}</p>
                                ${reminder.notes ? `<p><strong>Notes:</strong> ${reminder.notes}</p>` : ''}

                                <div class="reminder-times">
                                    <strong>Reminder Times:</strong>
                                    ${reminder.reminder_times.map(time => `<span class="time-badge">${time}</span>`).join('')}
                                </div>

                                ${reminder.last_taken ? `<p><strong>Last Taken:</strong> ${new Date(reminder.last_taken).toLocaleString()}</p>` : ''}
                                ${reminder.missed_doses > 0 ? `<p><strong>Missed Doses:</strong> ${reminder.missed_doses}</p>` : ''}
                            </div>
                        </div>
                    `;
                });

                container.innerHTML = html;
            }

            // Format frequency for display
            function formatFrequency(frequency) {
                const formats = {
                    'daily': 'Daily',
                    'twice_daily': 'Twice Daily',
                    'three_times_daily': 'Three Times Daily',
                    'weekly': 'Weekly',
                    'custom': 'Custom Schedule'
                };
                return formats[frequency] || frequency;
            }

            // Mark medication as taken
            async function markTaken(reminderId) {
                try {
                    const response = await fetch(`/medication-reminders/${reminderId}/mark-taken`, {
                        method: 'POST'
                    });

                    if (response.ok) {
                        showMessage('Medication marked as taken!', 'success');
                        loadReminders();
                    } else {
                        showMessage('Error marking medication as taken', 'error');
                    }
                } catch (error) {
                    showMessage('Error marking medication as taken: ' + error.message, 'error');
                }
            }

            // Toggle reminder enabled/disabled
            async function toggleReminder(reminderId, enabled) {
                try {
                    const reminder = await fetch(`/medication-reminders/${reminderId}`).then(r => r.json());
                    reminder.enabled = enabled;

                    const response = await fetch(`/medication-reminders/${reminderId}`, {
                        method: 'PUT',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(reminder)
                    });

                    if (response.ok) {
                        showMessage(`Reminder ${enabled ? 'enabled' : 'disabled'}!`, 'success');
                        loadReminders();
                    } else {
                        showMessage('Error updating reminder', 'error');
                    }
                } catch (error) {
                    showMessage('Error updating reminder: ' + error.message, 'error');
                }
            }

            // Delete reminder
            async function deleteReminder(reminderId) {
                if (!confirm('Are you sure you want to delete this reminder?')) return;

                try {
                    const response = await fetch(`/medication-reminders/${reminderId}`, {
                        method: 'DELETE'
                    });

                    if (response.ok) {
                        showMessage('Reminder deleted successfully!', 'success');
                        loadReminders();
                    } else {
                        showMessage('Error deleting reminder', 'error');
                    }
                } catch (error) {
                    showMessage('Error deleting reminder: ' + error.message, 'error');
                }
            }

            // Show message
            function showMessage(message, type) {
                const messagesDiv = document.getElementById('messages');
                const div = document.createElement('div');
                div.className = type === 'success' ? 'success-message' : 'error-message';
                div.textContent = message;
                messagesDiv.appendChild(div);

                setTimeout(() => {
                    div.remove();
                }, 5000);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/chat", response_class=HTMLResponse, tags=["ui"])
def ai_chatbot_interface():
    """AI chatbot interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MindTrack - AI Assistant</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .nav-links { text-align: center; margin: 20px 0; }
            .nav-links a { color: #2196F3; text-decoration: none; margin: 0 10px; }
            .nav-links a:hover { text-decoration: underline; }

            .chat-container { height: 400px; border: 1px solid #ddd; border-radius: 5px; padding: 20px; overflow-y: auto; background: #f9f9f9; margin-bottom: 20px; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; max-width: 80%; }
            .user-message { background: #e3f2fd; color: #1976d2; margin-left: auto; text-align: right; }
            .assistant-message { background: #f1f8e9; color: #33691e; }
            .message-time { font-size: 12px; color: #666; margin-top: 5px; }

            .input-container { display: flex; gap: 10px; }
            .message-input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
            .send-btn { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .send-btn:hover { background: #45a049; }
            .send-btn:disabled { background: #ccc; cursor: not-allowed; }

            .context-section { margin: 20px 0; padding: 20px; background: #e8f5e8; border-radius: 5px; }
            .context-form { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
            .form-group { margin: 10px 0; }
            .form-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
            .form-group input, .form-group select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }

            .typing-indicator { display: none; color: #666; font-style: italic; }

            .success-message { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0; }
            .error-message { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>MindTrack - AI Assistant</h1>
                <p>Chat with your AI assistant for mental health support, medication reminders, and personalized guidance.</p>
            </div>

            <div class="nav-links">
                <a href="/medication-reminders">Medication Reminders</a> | 
                <a href="/routines">Routines</a> | 
                <a href="/profile">Profile</a> | 
                <a href="/body-map">Body Map</a> | 
                <a href="/calendar">Calendar</a> | 
                <a href="/">Home</a>
            </div>

            <div id="messages"></div>

            <div class="context-section">
                <h3>Update Your Context</h3>
                <p>Help me provide better assistance by sharing your current state:</p>
                <div class="context-form">
                    <div class="form-group">
                        <label for="current-mood">Current Mood:</label>
                        <select id="current-mood">
                            <option value="">Select mood...</option>
                            <option value="very_positive">Very Positive</option>
                            <option value="positive">Positive</option>
                            <option value="neutral">Neutral</option>
                            <option value="negative">Negative</option>
                            <option value="very_negative">Very Negative</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="anxiety-level">Anxiety Level (0-10):</label>
                        <input type="number" id="anxiety-level" min="0" max="10" placeholder="0-10">
                    </div>
                    <div class="form-group">
                        <label for="sleep-quality">Sleep Quality:</label>
                        <select id="sleep-quality">
                            <option value="">Select quality...</option>
                            <option value="excellent">Excellent</option>
                            <option value="good">Good</option>
                            <option value="fair">Fair</option>
                            <option value="poor">Poor</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="current-symptoms">Current Symptoms:</label>
                        <input type="text" id="current-symptoms" placeholder="e.g., headache, fatigue, stress">
                    </div>
                </div>
                <button class="send-btn" onclick="updateContext()">Update Context</button>
            </div>

            <div class="chat-container" id="chatContainer">
                <div class="message assistant-message">
                    <div>Hello! I'm your MindTrack AI assistant. I'm here to help you with:</div>
                    <ul>
                        <li>Mental health support and guidance</li>
                        <li>Medication reminders and tracking</li>
                        <li>Symptom analysis and patterns</li>
                        <li>Routine suggestions and motivation</li>
                        <li>Emergency support when needed</li>
                    </ul>
                    <div>How can I help you today?</div>
                    <div class="message-time">Just now</div>
                </div>
            </div>

            <div class="typing-indicator" id="typingIndicator">AI is typing...</div>

            <div class="input-container">
                <input type="text" id="messageInput" class="message-input" placeholder="Type your message here..." onkeypress="handleKeyPress(event)">
                <button id="sendBtn" class="send-btn" onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            let sessionId = 'session_' + Date.now();
            let ws = null;

            // Initialize
            window.onload = function() {
                connectWebSocket();
            };

            // Connect to WebSocket
            function connectWebSocket() {
                ws = new WebSocket(`ws://${window.location.host}/chat/${sessionId}`);

                ws.onopen = function() {
                    console.log('WebSocket connected');
                };

                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.response) {
                        addMessage(data.response, 'assistant');
                        hideTypingIndicator();
                    } else if (data.error) {
                        addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
                        hideTypingIndicator();
                    }
                };

                ws.onclose = function() {
                    console.log('WebSocket disconnected');
                    setTimeout(connectWebSocket, 3000); // Reconnect after 3 seconds
                };

                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    addMessage('Connection error. Please refresh the page.', 'assistant');
                };
            }

            // Handle key press
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            // Send message
            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();

                if (!message) return;

                addMessage(message, 'user');
                input.value = '';

                showTypingIndicator();

                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ message: message }));
                } else {
                    addMessage('Connection lost. Please refresh the page.', 'assistant');
                    hideTypingIndicator();
                }
            }

            // Add message to chat
            function addMessage(content, sender) {
                const container = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;

                const contentDiv = document.createElement('div');
                contentDiv.innerHTML = content;
                messageDiv.appendChild(contentDiv);

                const timeDiv = document.createElement('div');
                timeDiv.className = 'message-time';
                timeDiv.textContent = new Date().toLocaleTimeString();
                messageDiv.appendChild(timeDiv);

                container.appendChild(messageDiv);
                container.scrollTop = container.scrollHeight;
            }

            // Show typing indicator
            function showTypingIndicator() {
                document.getElementById('typingIndicator').style.display = 'block';
                document.getElementById('sendBtn').disabled = true;
            }

            // Hide typing indicator
            function hideTypingIndicator() {
                document.getElementById('typingIndicator').style.display = 'none';
                document.getElementById('sendBtn').disabled = false;
            }

            // Update context
            async function updateContext() {
                const context = {
                    mood: document.getElementById('current-mood').value,
                    anxiety_level: parseInt(document.getElementById('anxiety-level').value) || 0,
                    sleep_quality: document.getElementById('sleep-quality').value,
                    symptoms: document.getElementById('current-symptoms').value
                };

                try {
                    const response = await fetch('/chat/update-context', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            session_id: sessionId,
                            context: context
                        })
                    });

                    if (response.ok) {
                        showMessage('Context updated successfully! I can now provide more personalized assistance.', 'success');
                    } else {
                        showMessage('Error updating context', 'error');
                    }
                } catch (error) {
                    showMessage('Error updating context: ' + error.message, 'error');
                }
            }

            // Show message
            function showMessage(message, type) {
                const messagesDiv = document.getElementById('messages');
                const div = document.createElement('div');
                div.className = type === 'success' ? 'success-message' : 'error-message';
                div.textContent = message;
                messagesDiv.appendChild(div);

                setTimeout(() => {
                    div.remove();
                }, 5000);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Main execution block
if __name__ == "__main__":
    import uvicorn

    print("Starting MindTrack server...")
    print("Open your browser and go to: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    uvicorn.run("Mindmap:app", host="0.0.0.0", port=8000, reload=True)
