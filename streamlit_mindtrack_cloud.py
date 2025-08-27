#!/usr/bin/env python3
"""
Streamlit Cloud-optimized MindTrack application.
This version is specifically designed for Streamlit Cloud deployment.
"""

import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
import time

# Configure Streamlit page
st.set_page_config(
    page_title="MindTrack - Mental Health Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = True  # Default to demo mode for cloud deployment

def show_demo_data():
    """Show demo data for the application."""
    demo_entries = [
        {"date": "2024-01-15", "mood": 8, "sleep_hours": 7.5, "energy": 7, "stress": 3},
        {"date": "2024-01-16", "mood": 6, "sleep_hours": 6.0, "energy": 5, "stress": 6},
        {"date": "2024-01-17", "mood": 9, "sleep_hours": 8.0, "energy": 8, "stress": 2},
        {"date": "2024-01-18", "mood": 7, "sleep_hours": 7.0, "energy": 6, "stress": 4},
        {"date": "2024-01-19", "mood": 8, "sleep_hours": 7.5, "energy": 7, "stress": 3},
    ]
    return pd.DataFrame(demo_entries)

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🧠 MindTrack - Mental Health Companion")
    st.markdown("Your comprehensive mental health and wellness tracking companion")
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["Dashboard", "Profile Management", "Body Map", "Calendar View", "Routines", "Medication Reminders", "Chat Assistant"]
        )
        
        st.divider()
        
        # Demo mode info
        st.info("🎮 **Demo Mode Active** - Using sample data for demonstration")
    
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
    
    # Get demo data
    data = show_demo_data()
    total_entries = len(data)
    this_week = len(data[data['date'] >= (date.today() - timedelta(days=7)).strftime('%Y-%m-%d')])
    current_streak = 3  # Demo value
    avg_sleep = data['sleep_hours'].mean()
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Entries", total_entries, "📝")
    
    with col2:
        st.metric("This Week", this_week, "📅")
    
    with col3:
        st.metric("Current Streak", current_streak, "🔥")
    
    with col4:
        st.metric("Avg Sleep", f"{avg_sleep:.1f}h", "😴")
    
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
    
    # Recent activity chart
    st.subheader("📈 Recent Activity")
    
    # Create simple chart using st.line_chart
    chart_data = data.set_index('date')[['mood', 'energy', 'stress']]
    st.line_chart(chart_data)
    
    # Data table
    st.subheader("📋 Recent Data")
    st.dataframe(data, use_container_width=True)

def show_profile_management():
    """Display profile management interface."""
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
            ["migraine", "anxiety", "depression", "insomnia", "chronic pain", "other"],
            default=["anxiety", "insomnia"]
        )
        
        symptoms = st.multiselect(
            "Recurring Symptoms",
            ["headache", "nausea", "fatigue", "dizziness", "mood swings", "other"],
            default=["headache", "fatigue"]
        )
        
        allergies = st.text_area("Allergies (one per line)", value="Penicillin\nPeanuts")
        
        submitted = st.form_submit_button("Save Profile")
        
        if submitted:
            st.success("✅ Profile saved successfully!")
            st.info("🎮 Demo mode: This is a demonstration. In a real app, this would save to the database.")

def show_body_map():
    """Display body map interface."""
    st.header("🗺️ Body Map")
    
    st.info("This feature provides an interactive body map for tracking symptoms and pain locations.")
    
    # Body map visualization placeholder
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image("https://via.placeholder.com/600x400?text=Interactive+Body+Map", 
                 caption="Interactive Body Map - Click to mark symptoms")
    
    with col2:
        st.subheader("Symptom Tracker")
        
        # Symptom selection
        symptom = st.selectbox("Select Symptom", ["headache", "nausea", "pain", "fatigue", "dizziness"])
        intensity = st.slider("Intensity (1-10)", 1, 10, 5)
        location = st.selectbox("Body Location", ["head", "neck", "chest", "stomach", "back", "arms", "legs"])
        
        if st.button("Add Symptom"):
            st.success(f"✅ Added {symptom} (intensity: {intensity}) to {location}")
            st.info("🎮 Demo mode: This is a demonstration.")

def show_calendar_view():
    """Display calendar view interface."""
    st.header("📅 Calendar View")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=date.today())
    
    # Generate demo data for the selected range
    dates = pd.date_range(start_date, end_date, freq='D')
    demo_data = pd.DataFrame({
        'date': dates,
        'mood': [7 + (i % 4) for i in range(len(dates))],
        'sleep_hours': [7.5 + (i % 2) for i in range(len(dates))],
        'energy': [6 + (i % 3) for i in range(len(dates))],
        'stress': [4 + (i % 3) for i in range(len(dates))]
    })
    
    # Create charts
    st.subheader("📊 Health Metrics Over Time")
    
    # Mood chart
    st.subheader("😊 Mood Trends")
    mood_chart = demo_data.set_index('date')['mood']
    st.line_chart(mood_chart)
    
    # Sleep chart
    st.subheader("😴 Sleep Hours")
    sleep_chart = demo_data.set_index('date')['sleep_hours']
    st.bar_chart(sleep_chart)
    
    # Data table
    st.subheader("📋 Detailed Data")
    st.dataframe(demo_data, use_container_width=True)

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
            start_time = st.time_input("Start Time", value=datetime.strptime("08:00", "%H:%M").time())
        with col2:
            end_time = st.time_input("End Time", value=datetime.strptime("09:00", "%H:%M").time())
        
        activities = st.text_area("Activities (one per line)", 
                                 value="Brush teeth\nTake medication\nExercise")
        notes = st.text_area("Notes", value="Important daily routine")
        
        submitted = st.form_submit_button("Add Routine")
        
        if submitted:
            st.success("✅ Routine added successfully!")
            st.info("🎮 Demo mode: This is a demonstration.")
    
    # Display existing routines
    st.subheader("📋 Current Routines")
    
    routines_data = {
        "Morning Routine": {"time": "08:00-09:00", "activities": ["Brush teeth", "Take medication", "Exercise"]},
        "Work Routine": {"time": "09:00-17:00", "activities": ["Check emails", "Team meeting", "Lunch break"]},
        "Evening Routine": {"time": "20:00-21:00", "activities": ["Read", "Meditation", "Prepare for bed"]}
    }
    
    for routine_name, routine_info in routines_data.items():
        with st.expander(f"🕐 {routine_name} ({routine_info['time']})"):
            for activity in routine_info['activities']:
                st.write(f"• {activity}")

def show_medication_reminders():
    """Display medication reminders interface."""
    st.header("💊 Medication Reminders")
    
    # Medication form
    with st.form("medication_form"):
        st.subheader("Add Medication")
        
        col1, col2 = st.columns(2)
        with col1:
            med_name = st.text_input("Medication Name", value="Vitamin D")
            dose = st.text_input("Dose", value="1000 IU")
        
        with col2:
            frequency = st.selectbox("Frequency", ["daily", "twice daily", "as needed"], index=0)
            time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening", "bedtime"], index=0)
        
        submitted = st.form_submit_button("Add Medication")
        
        if submitted:
            st.success("✅ Medication added successfully!")
            st.info("🎮 Demo mode: This is a demonstration.")
    
    # Display current medications
    st.subheader("💊 Current Medications")
    
    meds_data = [
        {"name": "Vitamin D", "dose": "1000 IU", "frequency": "daily", "time": "morning"},
        {"name": "Omega-3", "dose": "1000mg", "frequency": "daily", "time": "evening"},
        {"name": "Ibuprofen", "dose": "400mg", "frequency": "as needed", "time": "as needed"}
    ]
    
    for med in meds_data:
        with st.expander(f"💊 {med['name']} - {med['dose']}"):
            st.write(f"**Frequency:** {med['frequency']}")
            st.write(f"**Time:** {med['time']}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ Taken", key=f"taken_{med['name']}"):
                    st.success("Marked as taken!")
            with col2:
                if st.button(f"⏰ Remind Later", key=f"remind_{med['name']}"):
                    st.info("Reminder set for 1 hour later")

def show_chat_assistant():
    """Display chat assistant interface."""
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
                
                # Simple response logic
                if "mood" in prompt.lower():
                    response = "I notice you're asking about mood. Regular mood tracking can help identify patterns and triggers. Consider tracking your mood daily and noting what activities or events might be affecting it."
                elif "sleep" in prompt.lower():
                    response = "Sleep is crucial for mental health. Aim for 7-9 hours per night. Try establishing a consistent bedtime routine and avoid screens before bed."
                elif "stress" in prompt.lower():
                    response = "Stress management is important. Consider techniques like deep breathing, meditation, or regular exercise. What specific stressors are you dealing with?"
                elif "medication" in prompt.lower():
                    response = "Medication adherence is key for effectiveness. Set up reminders and track when you take your medications. Always consult with your healthcare provider about any concerns."
                else:
                    response = f"Thank you for your question about '{prompt}'. I'm here to help with your mental health journey. For personalized advice, consider tracking your symptoms and patterns regularly."
                
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
