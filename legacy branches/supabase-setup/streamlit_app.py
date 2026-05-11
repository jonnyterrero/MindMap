#!/usr/bin/env python3
"""
Run script for MindTrack Streamlit App
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting MindTrack Streamlit App...")
    print("=" * 50)
    print("📋 Features:")
    print("   • Dashboard with analytics")
    print("   • Daily mood and symptom logging")
    print("   • Personal profile management")
    print("   • Routine tracking and analytics")
    print("   • Calendar and streak tracking")
    print("   • Medication reminders")
    print("   • AI mental health assistant")
    print("=" * 50)
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 MindTrack app stopped.")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")
        print("💡 Make sure you have Streamlit installed:")
        print("   pip install streamlit pandas plotly")

if __name__ == "__main__":
    main()
