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
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
    symptoms = Column(JSON, nullable=True)          # list[str] - recurring symptoms
    allergies = Column(JSON, nullable=True)         # list[str] - allergies
    
    # Current Medications/Remedies
    current_medications = Column(JSON, nullable=True)  # list[dict] - {"name": "med_name", "dose": "10mg", "frequency": "daily"}
    supplements = Column(JSON, nullable=True)          # list[dict] - {"name": "vitamin_d", "dose": "1000iu", "frequency": "daily"}
    
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
    start_time = Column(String, nullable=True)     # 'HH:MM'
    end_time = Column(String, nullable=True)       # 'HH:MM'
    duration_minutes = Column(Integer, nullable=True)
    
    # Activities and Details
    activities = Column(JSON, nullable=True)       # list[str] - activities performed
    notes = Column(String, nullable=True)          # additional notes
    
    # Effectiveness Tracking
    symptoms_improved = Column(JSON, nullable=True)  # list[str] - symptoms that improved
    emotions_improved = Column(JSON, nullable=True)  # list[str] - emotions that improved
    sensations_improved = Column(JSON, nullable=True) # list[str] - body sensations that improved
    
    # Effectiveness Rating (0-10 scale)
    overall_effectiveness = Column(Integer, nullable=True)  # 0-10 scale
    energy_level_after = Column(Integer, nullable=True)     # 0-10 scale
    mood_improvement = Column(Integer, nullable=True)       # 0-10 scale
    symptom_relief = Column(Integer, nullable=True)         # 0-10 scale
    
    # Completion Status
    completed = Column(Boolean, default=True)
    completion_percentage = Column(Integer, nullable=True)  # 0-100
    
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
    sleep_hours = Column(Float, nullable=True)           # 0..24
    sleep_quality = Column(Integer, nullable=True)       # 1..5
    bed_time = Column(String, nullable=True)             # 'HH:MM'
    wake_time = Column(String, nullable=True)            # 'HH:MM'
    hrv = Column(Integer, nullable=True)                 # ms or index

    # Mental-health scales
    mood_valence = Column(Integer, nullable=True)        # -3..+3
    anxiety_level = Column(Integer, nullable=True)       # 0..10
    depression_level = Column(Integer, nullable=True)    # 0..10
    mania_level = Column(Integer, nullable=True)         # 0..10

    # ADHD / Focus / Productivity
    adhd_focus = Column(Integer, nullable=True)          # 0..10
    productivity_score = Column(Integer, nullable=True)  # 0..100

    # Routines
    routines_followed = Column(JSON, nullable=True)      # list[str]

    # Migraine
    migraine = Column(Boolean, nullable=False, default=False)
    migraine_intensity = Column(Integer, nullable=True)  # 0..10
    migraine_aura = Column(Boolean, nullable=True)

    # Body map sensations (JSON: {"head": 5, "chest": 3, "stomach": 7, ...})
    body_sensations = Column(JSON, nullable=True)        # dict[str, int] - body part -> intensity 0-10

    # Sensitive at-rest (optional encrypted)
    meds_cipher = Column(LargeBinary, nullable=True)     # bytes
    notes_cipher = Column(LargeBinary, nullable=True)    # bytes

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
    end_time: Optional[str] = None    # 'HH:MM'
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
        return {"date": today.isoformat(), "risk": "high", "reason": f"sleep_hours<{TRIGGER_SLEEP_HOURS_LT} and anxiety>={TRIGGER_ANXIETY_GE} yesterday"}
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


def get_intervention_recommendations(body_sensations: Dict[str, int], entries: List[Entry], user_profile: Optional[UserProfile] = None) -> dict:
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
            overall_recommendations = prioritize_user_preferences(overall_recommendations, user_profile.preferred_interventions)
        
        # Add condition-specific recommendations
        condition_recommendations = get_condition_specific_recommendations(user_profile, body_sensations)
        
        return {
            "body_part_recommendations": recommendations,
            "overall_recommendations": overall_recommendations[:5],
            "medication_suggestions": medication_suggestions,
            "condition_recommendations": condition_recommendations,
            "urgency_level": "high" if any(r["urgency"] == "high" for r in recommendations.values()) else "medium" if any(r["urgency"] == "medium" for r in recommendations.values()) else "low"
        }
    
    return {
        "body_part_recommendations": recommendations,
        "overall_recommendations": overall_recommendations[:5],
        "medication_suggestions": medication_suggestions,
        "urgency_level": "high" if any(r["urgency"] == "high" for r in recommendations.values()) else "medium" if any(r["urgency"] == "medium" for r in recommendations.values()) else "low"
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
            avg_sensation = sum(entry.body_sensations.values()) / len(entry.body_sensations) if entry.body_sensations else 0
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


def get_medication_suggestions(body_sensations: Dict[str, int], entries: List[Entry], user_profile: Optional[UserProfile] = None) -> List[dict]:
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
            "suggestions": [med["name"] for med in sorted(effective_meds, key=lambda x: x["effectiveness_score"], reverse=True)[:3]],
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
                return type.replace(/\b\w/g, l => l.toUpperCase());
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
                return intervention.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            }
            
            // Format body part names for display
            function formatBodyPartName(bodyPart) {
                return bodyPart.replace(/\b\w/g, l => l.toUpperCase());
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
