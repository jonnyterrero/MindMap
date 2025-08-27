#!/usr/bin/env python3
"""
Streamlit-compatible MindTrack application.
This file provides a Streamlit interface for the MindTrack features.
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(
    page_title="MindTrack - Mental Health Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://127.0.0.1:8000"

def check_api_connection():
    """Check if the FastAPI server is running."""
    try:
        response = requests.get(f"{st.session_state.api_base_url}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🧠 MindTrack - Mental Health Companion")
    st.markdown("Your comprehensive mental health and wellness tracking companion")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Dashboard", "Profile Management", "Body Map", "Calendar View", "Routines", "Medication Reminders", "Chat Assistant"]
    )
    
    # Check API connection
    if not check_api_connection():
        st.error("⚠️ Cannot connect to MindTrack API server. Please ensure the server is running.")
        st.info("To start the server, run: `python run_app.py` or `python test_app.py`")
        
        # API Configuration
        st.sidebar.subheader("API Configuration")
        new_url = st.sidebar.text_input("API Base URL", value=st.session_state.api_base_url)
        if st.sidebar.button("Update API URL"):
            st.session_state.api_base_url = new_url
            st.rerun()
        
        return
    
    # Main content based on selected page
    if page == "Dashboard":
        show_dashboard()
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
    elif page == "Chat Assistant":
        show_chat_assistant()

def show_dashboard():
    """Display the main dashboard."""
    st.header("📊 Dashboard")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Entries", "Loading...", "0")
    
    with col2:
        st.metric("This Week", "Loading...", "0")
    
    with col3:
        st.metric("Current Streak", "Loading...", "0")
    
    with col4:
        st.metric("Avg Sleep", "Loading...", "0h")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📝 Add Entry", use_container_width=True):
            st.info("Navigate to Profile Management to add entries")
    
    with col2:
        if st.button("🗺️ Body Map", use_container_width=True):
            st.info("Navigate to Body Map to view symptoms")
    
    with col3:
        if st.button("📅 Calendar", use_container_width=True):
            st.info("Navigate to Calendar View to see patterns")
    
    with col4:
        if st.button("💬 Chat", use_container_width=True):
            st.info("Navigate to Chat Assistant for help")

def show_profile_management():
    """Display profile management interface."""
    st.header("👤 Profile Management")
    
    # Profile form
    with st.form("profile_form"):
        st.subheader("Personal Information")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, max_value=120, value=25)
        
        with col2:
            gender = st.selectbox("Gender", ["", "male", "female", "non-binary", "other"])
            emergency_contact = st.text_input("Emergency Contact")
        
        st.subheader("Medical Information")
        known_conditions = st.multiselect(
            "Known Conditions",
            ["migraine", "anxiety", "depression", "insomnia", "chronic pain", "other"]
        )
        
        symptoms = st.multiselect(
            "Recurring Symptoms",
            ["headache", "nausea", "fatigue", "dizziness", "mood swings", "other"]
        )
        
        allergies = st.text_area("Allergies (one per line)")
        
        submitted = st.form_submit_button("Save Profile")
        
        if submitted:
            st.success("Profile saved successfully!")

def show_body_map():
    """Display body map interface."""
    st.header("🗺️ Body Map")
    
    st.info("This feature will show an interactive body map for tracking symptoms and pain locations.")
    
    # Placeholder for body map visualization
    st.image("https://via.placeholder.com/600x400?text=Body+Map+Visualization", 
             caption="Interactive Body Map - Coming Soon")

def show_calendar_view():
    """Display calendar view interface."""
    st.header("📅 Calendar View")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=date.today())
    
    # Placeholder for calendar heatmap
    st.info("This feature will show a calendar heatmap of your health data.")
    
    # Sample data for demonstration
    dates = pd.date_range(start_date, end_date, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'mood': [7 + (i % 3) for i in range(len(dates))],
        'sleep_hours': [7.5 + (i % 2) for i in range(len(dates))]
    })
    
    # Simple line chart
    fig = px.line(sample_data, x='date', y='mood', title='Mood Over Time')
    st.plotly_chart(fig, use_container_width=True)

def show_routines():
    """Display routines interface."""
    st.header("🔄 Routine Management")
    
    # Routine form
    with st.form("routine_form"):
        st.subheader("Add New Routine")
        
        routine_type = st.selectbox(
            "Routine Type",
            ["morning", "work", "school", "exercise", "afternoon", "evening"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.time_input("Start Time")
        with col2:
            end_time = st.time_input("End Time")
        
        activities = st.text_area("Activities (one per line)")
        notes = st.text_area("Notes")
        
        submitted = st.form_submit_button("Add Routine")
        
        if submitted:
            st.success("Routine added successfully!")

def show_medication_reminders():
    """Display medication reminders interface."""
    st.header("💊 Medication Reminders")
    
    # Medication form
    with st.form("medication_form"):
        st.subheader("Add Medication")
        
        col1, col2 = st.columns(2)
        with col1:
            med_name = st.text_input("Medication Name")
            dose = st.text_input("Dose")
        
        with col2:
            frequency = st.selectbox("Frequency", ["daily", "twice daily", "as needed"])
            time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening", "bedtime"])
        
        submitted = st.form_submit_button("Add Medication")
        
        if submitted:
            st.success("Medication added successfully!")

def show_chat_assistant():
    """Display chat assistant interface."""
    st.header("💬 AI Chat Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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
        
        # Display assistant response
        with st.chat_message("assistant"):
            response = f"Thank you for your question: '{prompt}'. I'm here to help with your mental health journey. This is a placeholder response - the full AI chat functionality will be integrated with your FastAPI backend."
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
