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
if 'body_symptoms' not in st.session_state:
    st.session_state.body_symptoms = {}

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
    
    # Summary statistics
    if st.session_state.body_symptoms:
        st.subheader("📈 Symptom Summary")
        
        total_symptoms = sum(len(symptoms) for symptoms in st.session_state.body_symptoms.values())
        avg_intensity = 0
        symptom_count = 0
        
        for symptoms in st.session_state.body_symptoms.values():
            for symptom in symptoms:
                avg_intensity += symptom['intensity']
                symptom_count += 1
        
        if symptom_count > 0:
            avg_intensity = avg_intensity / symptom_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Symptoms", total_symptoms)
        with col2:
            st.metric("Affected Regions", len(st.session_state.body_symptoms))
        with col3:
            st.metric("Avg Intensity", f"{avg_intensity:.1f}/10")
        
        # Most common symptoms
        symptom_types = {}
        for symptoms in st.session_state.body_symptoms.values():
            for symptom in symptoms:
                symptom_type = symptom['type']
                symptom_types[symptom_type] = symptom_types.get(symptom_type, 0) + 1
        
        if symptom_types:
            st.write("**Most Common Symptoms:**")
            for symptom_type, count in sorted(symptom_types.items(), key=lambda x: x[1], reverse=True)[:3]:
                st.write(f"• {symptom_type.title()}: {count} occurrences")

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
