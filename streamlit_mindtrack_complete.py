#!/usr/bin/env python3
"""
Comprehensive MindTrack Streamlit Application
Integrates all features from the FastAPI version into a single Streamlit app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import time
import json
import io
import base64
from typing import Optional, Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure Streamlit page
st.set_page_config(
    page_title="MindTrack - Complete Mental Health Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for all features
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = True
if 'body_symptoms' not in st.session_state:
    st.session_state.body_symptoms = {}
if 'entries' not in st.session_state:
    st.session_state.entries = []
if 'routines' not in st.session_state:
    st.session_state.routines = []
if 'medications' not in st.session_state:
    st.session_state.medications = []
if 'emergency_alerts' not in st.session_state:
    st.session_state.emergency_alerts = []
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = []

def generate_demo_data():
    """Generate comprehensive demo data for all features."""
    # Generate entries for the last 30 days
    entries = []
    for i in range(30):
        entry_date = date.today() - timedelta(days=29-i)
        entries.append({
            'id': i + 1,
            'date': entry_date,
            'sleep_hours': 7.5 + np.random.normal(0, 1),
            'sleep_quality': np.random.randint(1, 6),
            'mood_valence': np.random.randint(-3, 4),
            'anxiety_level': np.random.randint(0, 11),
            'depression_level': np.random.randint(0, 11),
            'adhd_focus': np.random.randint(0, 11),
            'productivity_score': np.random.randint(0, 101),
            'migraine': np.random.choice([True, False], p=[0.2, 0.8]),
            'migraine_intensity': np.random.randint(0, 11) if np.random.choice([True, False], p=[0.2, 0.8]) else None,
            'body_sensations': {
                'head': np.random.randint(0, 11),
                'neck': np.random.randint(0, 11),
                'chest': np.random.randint(0, 11),
                'stomach': np.random.randint(0, 11),
                'back': np.random.randint(0, 11)
            },
            'notes': f"Demo entry for {entry_date.strftime('%Y-%m-%d')}"
        })
    
    # Generate routines
    routines = [
        {
            'id': 1,
            'routine_type': 'morning',
            'start_time': '08:00',
            'end_time': '09:00',
            'activities': ['Brush teeth', 'Take medication', 'Exercise'],
            'notes': 'Important morning routine',
            'overall_effectiveness': 8,
            'completed': True
        },
        {
            'id': 2,
            'routine_type': 'evening',
            'start_time': '20:00',
            'end_time': '21:00',
            'activities': ['Read', 'Meditation', 'Prepare for bed'],
            'notes': 'Evening wind-down routine',
            'overall_effectiveness': 7,
            'completed': True
        }
    ]
    
    # Generate medications
    medications = [
        {
            'id': 1,
            'medication_name': 'Vitamin D',
            'dose': '1000 IU',
            'frequency': 'daily',
            'reminder_times': ['08:00'],
            'enabled': True,
            'last_taken': datetime.now() - timedelta(hours=12),
            'next_reminder': datetime.now() + timedelta(hours=12),
            'missed_doses': 0
        },
        {
            'id': 2,
            'medication_name': 'Omega-3',
            'dose': '1000mg',
            'frequency': 'daily',
            'reminder_times': ['20:00'],
            'enabled': True,
            'last_taken': datetime.now() - timedelta(hours=2),
            'next_reminder': datetime.now() + timedelta(hours=22),
            'missed_doses': 1
        }
    ]
    
    # Generate weather data
    weather_data = []
    for i in range(30):
        weather_date = date.today() - timedelta(days=29-i)
        weather_data.append({
            'date': weather_date,
            'temperature_high': 20 + np.random.normal(0, 5),
            'temperature_low': 10 + np.random.normal(0, 5),
            'humidity': np.random.randint(30, 90),
            'precipitation': np.random.choice([0, 5, 10, 20], p=[0.7, 0.1, 0.1, 0.1]),
            'weather_condition': np.random.choice(['sunny', 'cloudy', 'rainy', 'partly_cloudy'])
        })
    
    return entries, routines, medications, weather_data

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🧠 MindTrack - Complete Mental Health Companion")
    st.markdown("Your comprehensive mental health and wellness tracking companion with advanced analytics")
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.selectbox(
            "Choose a page:",
            [
                "Dashboard", 
                "Profile Management", 
                "Body Map", 
                "Calendar View", 
                "Routines", 
                "Medication Reminders",
                "Emergency Alerts",
                "Weather Correlation",
                "Advanced Analytics",
                "Data Export",
                "Chat Assistant"
            ]
        )
        
        st.divider()
        
        # Demo mode info
        st.info("🎮 **Demo Mode Active** - Using sample data for demonstration")
        
        # Quick stats
        if st.session_state.demo_mode:
            entries, routines, medications, weather_data = generate_demo_data()
            st.subheader("📊 Quick Stats")
            st.metric("Total Entries", len(entries))
            st.metric("Active Routines", len(routines))
            st.metric("Medications", len(medications))
            st.metric("Weather Days", len(weather_data))
    
    # Main content based on selected page
    if page == "Dashboard":
        show_enhanced_dashboard()
    elif page == "Profile Management":
        show_profile_management()
    elif page == "Body Map":
        show_body_map()
    elif page == "Calendar View":
        show_calendar_view()
    elif page == "Routines":
        show_routines()
    elif page == "Medication Reminders":
        show_medication_reminders()
    elif page == "Emergency Alerts":
        show_emergency_alerts()
    elif page == "Weather Correlation":
        show_weather_correlation()
    elif page == "Advanced Analytics":
        show_advanced_analytics()
    elif page == "Data Export":
        show_data_export()
    elif page == "Chat Assistant":
        show_chat_assistant()

def show_enhanced_dashboard():
    """Display enhanced dashboard with comprehensive metrics."""
    st.header("📊 Enhanced Dashboard")
    
    # Generate demo data
    entries, routines, medications, weather_data = generate_demo_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_sleep = np.mean([e['sleep_hours'] for e in entries if e['sleep_hours']])
        st.metric("Avg Sleep", f"{avg_sleep:.1f}h", "😴")
    
    with col2:
        avg_mood = np.mean([e['mood_valence'] for e in entries if e['mood_valence'] is not None])
        st.metric("Avg Mood", f"{avg_mood:.1f}", "😊")
    
    with col3:
        migraine_days = sum(1 for e in entries if e['migraine'])
        st.metric("Migraine Days", migraine_days, "🤕")
    
    with col4:
        avg_anxiety = np.mean([e['anxiety_level'] for e in entries if e['anxiety_level'] is not None])
        st.metric("Avg Anxiety", f"{avg_anxiety:.1f}/10", "😰")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sleep Trends")
        sleep_data = pd.DataFrame([{
            'date': e['date'],
            'sleep_hours': e['sleep_hours'],
            'sleep_quality': e['sleep_quality']
        } for e in entries])
        
        fig = px.line(sleep_data, x='date', y='sleep_hours', 
                     title='Sleep Hours Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("😊 Mood & Anxiety")
        mood_data = pd.DataFrame([{
            'date': e['date'],
            'mood': e['mood_valence'],
            'anxiety': e['anxiety_level']
        } for e in entries])
        
        fig = px.line(mood_data, x='date', y=['mood', 'anxiety'], 
                     title='Mood and Anxiety Trends')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent entries table
    st.subheader("📋 Recent Entries")
    recent_entries = pd.DataFrame(entries[-10:])
    st.dataframe(recent_entries[['date', 'sleep_hours', 'mood_valence', 'anxiety_level', 'migraine']], 
                use_container_width=True)

def show_profile_management():
    """Display enhanced profile management interface."""
    st.header("👤 Profile Management")
    
    # Profile form
    with st.form("profile_form"):
        st.subheader("Personal Information")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value="John Doe")
            age = st.number_input("Age", min_value=0, max_value=120, value=25)
        
        with col2:
            gender = st.selectbox("Gender", ["", "male", "female", "non-binary", "other"])
            emergency_contact = st.text_input("Emergency Contact", value="+1-555-0123")
        
        st.subheader("Medical Information")
        known_conditions = st.multiselect(
            "Known Conditions",
            ["migraine", "anxiety", "depression", "insomnia", "chronic pain", "ADHD", "bipolar", "PTSD", "other"],
            default=["anxiety", "insomnia"]
        )
        
        symptoms = st.multiselect(
            "Recurring Symptoms",
            ["headache", "nausea", "fatigue", "dizziness", "mood swings", "panic attacks", "racing thoughts", "other"],
            default=["headache", "fatigue"]
        )
        
        allergies = st.text_area("Allergies (one per line)", value="Penicillin\nPeanuts")
        
        st.subheader("Current Medications")
        med_name = st.text_input("Medication Name", value="Vitamin D")
        med_dose = st.text_input("Dose", value="1000 IU")
        med_frequency = st.selectbox("Frequency", ["daily", "twice daily", "as needed"], index=0)
        
        submitted = st.form_submit_button("Save Profile")
        
        if submitted:
            st.success("✅ Profile saved successfully!")
            st.info("🎮 Demo mode: This is a demonstration.")

def show_body_map():
    """Display interactive body map interface."""
    st.header("🗺️ Interactive Body Map")
    
    st.info("Click on different body regions to track symptoms and pain locations.")
    
    # Body map layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("👤 Click on Body Regions")
        
        # Create interactive body map using buttons arranged in body shape
        # Head
        if st.button("🧠 Head", key="head", use_container_width=True):
            st.session_state.selected_region = "head"
            st.rerun()
        
        # Neck
        if st.button("👔 Neck", key="neck", use_container_width=True):
            st.session_state.selected_region = "neck"
            st.rerun()
        
        # Shoulders
        col_shoulder1, col_shoulder2 = st.columns(2)
        with col_shoulder1:
            if st.button("💪 Left Shoulder", key="left_shoulder"):
                st.session_state.selected_region = "left_shoulder"
                st.rerun()
        with col_shoulder2:
            if st.button("💪 Right Shoulder", key="right_shoulder"):
                st.session_state.selected_region = "right_shoulder"
                st.rerun()
        
        # Arms
        col_arm1, col_arm2 = st.columns(2)
        with col_arm1:
            if st.button("🦾 Left Arm", key="left_arm"):
                st.session_state.selected_region = "left_arm"
                st.rerun()
        with col_arm2:
            if st.button("🦾 Right Arm", key="right_arm"):
                st.session_state.selected_region = "right_arm"
                st.rerun()
        
        # Chest
        if st.button("🫁 Chest", key="chest", use_container_width=True):
            st.session_state.selected_region = "chest"
            st.rerun()
        
        # Stomach
        if st.button("🤰 Stomach", key="stomach", use_container_width=True):
            st.session_state.selected_region = "stomach"
            st.rerun()
        
        # Back
        if st.button("🫂 Back", key="back", use_container_width=True):
            st.session_state.selected_region = "back"
            st.rerun()
        
        # Legs
        col_leg1, col_leg2 = st.columns(2)
        with col_leg1:
            if st.button("🦵 Left Leg", key="left_leg"):
                st.session_state.selected_region = "left_leg"
                st.rerun()
        with col_leg2:
            if st.button("🦵 Right Leg", key="right_leg"):
                st.session_state.selected_region = "right_leg"
                st.rerun()
        
        # Feet
        col_foot1, col_foot2 = st.columns(2)
        with col_foot1:
            if st.button("🦶 Left Foot", key="left_foot"):
                st.session_state.selected_region = "left_foot"
                st.rerun()
        with col_foot2:
            if st.button("🦶 Right Foot", key="right_foot"):
                st.session_state.selected_region = "right_foot"
                st.rerun()
    
    with col2:
        st.subheader("📝 Symptom Tracker")
        
        # Show selected region
        if 'selected_region' in st.session_state:
            st.success(f"Selected: **{st.session_state.selected_region.replace('_', ' ').title()}**")
            
            # Symptom form for selected region
            with st.form(f"symptom_form_{st.session_state.selected_region}"):
                symptom_type = st.selectbox(
                    "Symptom Type",
                    ["pain", "headache", "nausea", "fatigue", "dizziness", "swelling", "itching", "burning", "numbness", "other"],
                    key=f"symptom_type_{st.session_state.selected_region}"
                )
                
                intensity = st.slider("Intensity (1-10)", 1, 10, 5, key=f"intensity_{st.session_state.selected_region}")
                
                duration = st.selectbox(
                    "Duration",
                    ["just started", "few minutes", "few hours", "all day", "few days", "week+"],
                    key=f"duration_{st.session_state.selected_region}"
                )
                
                notes = st.text_area("Notes", key=f"notes_{st.session_state.selected_region}")
                
                col_add, col_clear = st.columns(2)
                with col_add:
                    if st.form_submit_button("➕ Add Symptom"):
                        # Add symptom to session state
                        if st.session_state.selected_region not in st.session_state.body_symptoms:
                            st.session_state.body_symptoms[st.session_state.selected_region] = []
                        
                        new_symptom = {
                            "type": symptom_type,
                            "intensity": intensity,
                            "duration": duration,
                            "notes": notes,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                        
                        st.session_state.body_symptoms[st.session_state.selected_region].append(new_symptom)
                        st.success(f"✅ Added {symptom_type} to {st.session_state.selected_region}")
                        st.rerun()
                
                with col_clear:
                    if st.form_submit_button("🗑️ Clear Region"):
                        if st.session_state.selected_region in st.session_state.body_symptoms:
                            del st.session_state.body_symptoms[st.session_state.selected_region]
                        st.success(f"✅ Cleared symptoms from {st.session_state.selected_region}")
                        st.rerun()
        else:
            st.info("👆 Click on a body region above to start tracking symptoms")
    
    # Display current symptoms
    if st.session_state.body_symptoms:
        st.subheader("📊 Current Symptoms")
        
        for region, symptoms in st.session_state.body_symptoms.items():
            with st.expander(f"📍 {region.replace('_', ' ').title()} ({len(symptoms)} symptoms)"):
                for i, symptom in enumerate(symptoms):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{symptom['type'].title()}** - Intensity: {symptom['intensity']}/10")
                        st.write(f"Duration: {symptom['duration']}")
                        if symptom['notes']:
                            st.write(f"Notes: {symptom['notes']}")
                    with col2:
                        st.write(f"Added: {symptom['timestamp']}")
                    with col3:
                        if st.button(f"❌ Remove", key=f"remove_{region}_{i}"):
                            st.session_state.body_symptoms[region].pop(i)
                            if not st.session_state.body_symptoms[region]:
                                del st.session_state.body_symptoms[region]
                            st.rerun()
                    st.divider()

def show_calendar_view():
    """Display enhanced calendar view with heatmaps."""
    st.header("📅 Calendar View")
    
    # Generate demo data
    entries, _, _, _ = generate_demo_data()
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=date.today())
    
    # Create calendar heatmap
    st.subheader("🌡️ Sleep Quality Heatmap")
    
    # Prepare data for heatmap
    calendar_data = []
    for entry in entries:
        if start_date <= entry['date'] <= end_date:
            calendar_data.append({
                'date': entry['date'],
                'sleep_hours': entry['sleep_hours'],
                'sleep_quality': entry['sleep_quality'],
                'mood': entry['mood_valence'],
                'anxiety': entry['anxiety_level']
            })
    
    if calendar_data:
        df = pd.DataFrame(calendar_data)
        
        # Sleep quality heatmap
        fig = px.imshow(
            df.pivot_table(index=df['date'].dt.day, columns=df['date'].dt.month, values='sleep_quality', aggfunc='mean'),
            title='Sleep Quality Heatmap',
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Mood heatmap
        st.subheader("😊 Mood Heatmap")
        fig2 = px.imshow(
            df.pivot_table(index=df['date'].dt.day, columns=df['date'].dt.month, values='mood', aggfunc='mean'),
            title='Mood Heatmap',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig2, use_container_width=True)

def show_routines():
    """Display enhanced routine management."""
    st.header("🔄 Routine Management")
    
    # Generate demo data
    _, routines, _, _ = generate_demo_data()
    
    # Add new routine
    with st.form("routine_form"):
        st.subheader("Add New Routine")
        
        routine_type = st.selectbox(
            "Routine Type",
            ["morning", "work", "school", "exercise", "afternoon", "evening"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.time_input("Start Time", value=datetime.strptime("08:00", "%H:%M").time())
        with col2:
            end_time = st.time_input("End Time", value=datetime.strptime("09:00", "%H:%M").time())
        
        activities = st.text_area("Activities (one per line)", 
                                 value="Brush teeth\nTake medication\nExercise")
        notes = st.text_area("Notes", value="Important daily routine")
        
        effectiveness = st.slider("Expected Effectiveness (1-10)", 1, 10, 7)
        
        submitted = st.form_submit_button("Add Routine")
        
        if submitted:
            st.success("✅ Routine added successfully!")
            st.info("🎮 Demo mode: This is a demonstration.")
    
    # Display existing routines
    st.subheader("📋 Current Routines")
    
    for routine in routines:
        with st.expander(f"🕐 {routine['routine_type'].title()} Routine ({routine['start_time']}-{routine['end_time']})"):
            st.write(f"**Activities:**")
            for activity in routine['activities']:
                st.write(f"• {activity}")
            st.write(f"**Notes:** {routine['notes']}")
            st.write(f"**Effectiveness:** {routine['overall_effectiveness']}/10")
            st.write(f"**Status:** {'✅ Completed' if routine['completed'] else '⏳ Pending'}")

def show_medication_reminders():
    """Display enhanced medication reminders."""
    st.header("💊 Medication Reminders")
    
    # Generate demo data
    _, _, medications, _ = generate_demo_data()
    
    # Add new medication
    with st.form("medication_form"):
        st.subheader("Add New Medication")
        
        col1, col2 = st.columns(2)
        with col1:
            med_name = st.text_input("Medication Name", value="Vitamin D")
            dose = st.text_input("Dose", value="1000 IU")
        
        with col2:
            frequency = st.selectbox("Frequency", ["daily", "twice daily", "three times daily", "weekly", "as needed"], index=0)
            reminder_time = st.time_input("Reminder Time", value=datetime.strptime("08:00", "%H:%M").time())
        
        enabled = st.checkbox("Enable Reminders", value=True)
        notes = st.text_area("Notes")
        
        submitted = st.form_submit_button("Add Medication")
        
        if submitted:
            st.success("✅ Medication added successfully!")
            st.info("🎮 Demo mode: This is a demonstration.")
    
    # Display current medications
    st.subheader("💊 Current Medications")
    
    for med in medications:
        with st.expander(f"💊 {med['medication_name']} - {med['dose']}"):
            st.write(f"**Frequency:** {med['frequency']}")
            st.write(f"**Reminder Times:** {', '.join(med['reminder_times'])}")
            st.write(f"**Status:** {'✅ Enabled' if med['enabled'] else '❌ Disabled'}")
            st.write(f"**Last Taken:** {med['last_taken'].strftime('%Y-%m-%d %H:%M') if med['last_taken'] else 'Never'}")
            st.write(f"**Next Reminder:** {med['next_reminder'].strftime('%Y-%m-%d %H:%M') if med['next_reminder'] else 'Not set'}")
            st.write(f"**Missed Doses:** {med['missed_doses']}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ Mark Taken", key=f"taken_{med['id']}"):
                    st.success("Marked as taken!")
            with col2:
                if st.button(f"⏰ Remind Later", key=f"remind_{med['id']}"):
                    st.info("Reminder set for 1 hour later")

def show_emergency_alerts():
    """Display emergency alerts system."""
    st.header("🚨 Emergency Alerts")
    
    st.warning("⚠️ This feature is for demonstration purposes. In a real app, this would connect to emergency services.")
    
    # Alert types
    alert_types = {
        'high_anxiety': 'High Anxiety',
        'suicidal_thoughts': 'Suicidal Thoughts',
        'severe_migraine': 'Severe Migraine',
        'medication_overdose': 'Medication Overdose',
        'panic_attack': 'Panic Attack'
    }
    
    # Create alert
    with st.form("alert_form"):
        st.subheader("Create Emergency Alert")
        
        alert_type = st.selectbox("Alert Type", list(alert_types.keys()), format_func=lambda x: alert_types[x])
        severity = st.selectbox("Severity", ["low", "medium", "high", "critical"])
        trigger_data = st.text_area("What triggered this alert?")
        user_notes = st.text_area("Additional notes")
        
        submitted = st.form_submit_button("🚨 Create Alert")
        
        if submitted:
            st.error("🚨 Emergency Alert Created!")
            st.info("🎮 Demo mode: This is a demonstration.")
    
    # Demo alerts
    st.subheader("📋 Recent Alerts")
    
    demo_alerts = [
        {
            'id': 1,
            'alert_type': 'high_anxiety',
            'severity': 'medium',
            'triggered_by': 'Work stress',
            'status': 'resolved',
            'created_at': datetime.now() - timedelta(days=2)
        },
        {
            'id': 2,
            'alert_type': 'severe_migraine',
            'severity': 'high',
            'triggered_by': 'Weather change',
            'status': 'active',
            'created_at': datetime.now() - timedelta(hours=6)
        }
    ]
    
    for alert in demo_alerts:
        with st.expander(f"🚨 {alert_types[alert['alert_type']]} - {alert['severity'].upper()}"):
            st.write(f"**Triggered by:** {alert['triggered_by']}")
            st.write(f"**Status:** {alert['status'].title()}")
            st.write(f"**Created:** {alert['created_at'].strftime('%Y-%m-%d %H:%M')}")
            
            if alert['status'] == 'active':
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✅ Acknowledge", key=f"ack_{alert['id']}"):
                        st.success("Alert acknowledged!")
                with col2:
                    if st.button(f"🔒 Resolve", key=f"resolve_{alert['id']}"):
                        st.success("Alert resolved!")

def show_weather_correlation():
    """Display weather correlation analysis."""
    st.header("🌤️ Weather Correlation")
    
    # Generate demo data
    entries, _, _, weather_data = generate_demo_data()
    
    st.info("Analyze how weather conditions affect your symptoms and mood.")
    
    # Weather data
    st.subheader("📊 Weather Data")
    
    weather_df = pd.DataFrame(weather_data)
    st.dataframe(weather_df, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Symptom-Weather Correlation")
    
    # Create correlation data
    correlation_data = []
    for entry in entries:
        weather_entry = next((w for w in weather_data if w['date'] == entry['date']), None)
        if weather_entry:
            correlation_data.append({
                'date': entry['date'],
                'temperature': weather_entry['temperature_high'],
                'humidity': weather_entry['humidity'],
                'precipitation': weather_entry['precipitation'],
                'migraine': entry['migraine'],
                'anxiety': entry['anxiety_level'],
                'mood': entry['mood_valence']
            })
    
    if correlation_data:
        corr_df = pd.DataFrame(correlation_data)
        
        # Correlation matrix
        numeric_cols = ['temperature', 'humidity', 'precipitation', 'anxiety', 'mood']
        correlation_matrix = corr_df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title='Weather-Symptom Correlation Matrix',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Migraine vs weather
        st.subheader("🤕 Migraine vs Weather")
        migraine_data = corr_df[corr_df['migraine'] == True]
        if not migraine_data.empty:
            fig2 = px.scatter(
                migraine_data, 
                x='temperature', 
                y='humidity',
                title='Migraine Occurrences vs Temperature & Humidity',
                color='anxiety',
                size='precipitation'
            )
            st.plotly_chart(fig2, use_container_width=True)

def show_advanced_analytics():
    """Display advanced analytics and insights."""
    st.header("📈 Advanced Analytics")
    
    # Generate demo data
    entries, routines, medications, weather_data = generate_demo_data()
    
    # Sleep analysis
    st.subheader("😴 Sleep Analysis")
    
    sleep_data = pd.DataFrame([{
        'date': e['date'],
        'sleep_hours': e['sleep_hours'],
        'sleep_quality': e['sleep_quality'],
        'mood': e['mood_valence'],
        'anxiety': e['anxiety_level']
    } for e in entries])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sleep vs mood correlation
        fig = px.scatter(
            sleep_data, 
            x='sleep_hours', 
            y='mood',
            title='Sleep Hours vs Mood',
            trendline='ols'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sleep quality distribution
        fig2 = px.histogram(
            sleep_data, 
            x='sleep_quality',
            title='Sleep Quality Distribution',
            nbins=5
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Trigger analysis
    st.subheader("⚠️ Trigger Analysis")
    
    # Identify potential triggers
    high_anxiety_days = [e for e in entries if e['anxiety_level'] and e['anxiety_level'] >= 7]
    migraine_days = [e for e in entries if e['migraine']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("High Anxiety Days", len(high_anxiety_days))
        if high_anxiety_days:
            avg_sleep_anxiety = np.mean([e['sleep_hours'] for e in high_anxiety_days if e['sleep_hours']])
            st.metric("Avg Sleep on High Anxiety Days", f"{avg_sleep_anxiety:.1f}h")
    
    with col2:
        st.metric("Migraine Days", len(migraine_days))
        if migraine_days:
            avg_sleep_migraine = np.mean([e['sleep_hours'] for e in migraine_days if e['sleep_hours']])
            st.metric("Avg Sleep on Migraine Days", f"{avg_sleep_migraine:.1f}h")
    
    # Streak analysis
    st.subheader("🔥 Streak Analysis")
    
    # Calculate streaks
    good_sleep_streak = 0
    current_streak = 0
    
    for entry in reversed(entries):
        if entry['sleep_hours'] and entry['sleep_hours'] >= 7:
            current_streak += 1
        else:
            break
    
    good_sleep_streak = current_streak
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Good Sleep Streak", good_sleep_streak, "🔥")
    
    with col2:
        total_entries = len(entries)
        st.metric("Total Tracking Days", total_entries, "📅")
    
    with col3:
        completion_rate = (total_entries / 30) * 100
        st.metric("Tracking Completion", f"{completion_rate:.1f}%", "✅")

def show_data_export():
    """Display data export functionality."""
    st.header("📤 Data Export")
    
    # Generate demo data
    entries, routines, medications, weather_data = generate_demo_data()
    
    st.info("Export your data in various formats for backup or analysis.")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Export Options")
        
        if st.button("📄 Export as CSV"):
            # Create CSV data
            entries_df = pd.DataFrame(entries)
            csv = entries_df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"mindtrack_data_{date.today()}.csv",
                mime="text/csv"
            )
        
        if st.button("📋 Export as JSON"):
            # Create JSON data
            export_data = {
                'entries': entries,
                'routines': routines,
                'medications': medications,
                'weather_data': weather_data,
                'export_date': datetime.now().isoformat()
            }
            
            json_data = json.dumps(export_data, indent=2, default=str)
            
            st.download_button(
                label="⬇️ Download JSON",
                data=json_data,
                file_name=f"mindtrack_data_{date.today()}.json",
                mime="application/json"
            )
    
    with col2:
        st.subheader("📈 Data Summary")
        
        st.write(f"**Total Entries:** {len(entries)}")
        st.write(f"**Total Routines:** {len(routines)}")
        st.write(f"**Total Medications:** {len(medications)}")
        st.write(f"**Weather Data Points:** {len(weather_data)}")
        
        # Data preview
        st.subheader("👀 Data Preview")
        
        preview_option = st.selectbox("Select data to preview:", ["Entries", "Routines", "Medications", "Weather"])
        
        if preview_option == "Entries":
            st.dataframe(pd.DataFrame(entries).head(), use_container_width=True)
        elif preview_option == "Routines":
            st.dataframe(pd.DataFrame(routines).head(), use_container_width=True)
        elif preview_option == "Medications":
            st.dataframe(pd.DataFrame(medications).head(), use_container_width=True)
        elif preview_option == "Weather":
            st.dataframe(pd.DataFrame(weather_data).head(), use_container_width=True)

def show_chat_assistant():
    """Display enhanced chat assistant."""
    st.header("💬 AI Chat Assistant")
    
    st.info("Get personalized insights, recommendations, and answers to your health questions.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your health..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                time.sleep(1)  # Simulate processing time
                
                # Enhanced response logic
                if "mood" in prompt.lower():
                    response = "I notice you're asking about mood. Regular mood tracking can help identify patterns and triggers. Consider tracking your mood daily and noting what activities or events might be affecting it. Based on your data, I can see patterns in your mood fluctuations."
                elif "sleep" in prompt.lower():
                    response = "Sleep is crucial for mental health. Aim for 7-9 hours per night. Try establishing a consistent bedtime routine and avoid screens before bed. Your sleep data shows some variability - consider tracking what affects your sleep quality."
                elif "stress" in prompt.lower() or "anxiety" in prompt.lower():
                    response = "Stress management is important. Consider techniques like deep breathing, meditation, or regular exercise. What specific stressors are you dealing with? I can help you identify patterns in your anxiety levels."
                elif "medication" in prompt.lower():
                    response = "Medication adherence is key for effectiveness. Set up reminders and track when you take your medications. Always consult with your healthcare provider about any concerns. Your medication tracking shows good consistency."
                elif "migraine" in prompt.lower():
                    response = "Migraine tracking is important for identifying triggers. Consider tracking weather changes, stress levels, and sleep patterns. Your data shows some correlation between certain factors and migraine occurrences."
                elif "routine" in prompt.lower():
                    response = "Routines can significantly improve mental health. Try to establish consistent daily patterns, especially for sleep, meals, and exercise. Your routine data shows good completion rates."
                else:
                    response = f"Thank you for your question about '{prompt}'. I'm here to help with your mental health journey. For personalized advice, consider tracking your symptoms and patterns regularly. I can analyze your data to provide specific insights."
                
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
