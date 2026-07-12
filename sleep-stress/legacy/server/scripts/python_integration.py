#!/usr/bin/env python3
"""
Python Integration Script for Sleep & Stress + App
This script connects the Node.js backend with your Python analysis system
"""

import os
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
import argparse

# Configuration
API_BASE_URL = "http://localhost:3000/api"
DATA_FILE = "health_logs.json"

class SleepStressPlusIntegration:
    def __init__(self, api_base_url=API_BASE_URL):
        self.api_base_url = api_base_url
        self.data_file = DATA_FILE
    
    def init_dataset(self):
        """Initialize data storage (matches your Python function)"""
        if not os.path.exists(self.data_file):
            with open(self.data_file, "w") as f:
                pass  # create empty file
    
    def log_entry(self, date, sleep_start, sleep_end, quality, mood_score, stress_score,
                  journal="", voice_note=None, gi_flare=0, skin_flare=0, migraine=0):
        """Log entry (matches your Python function)"""
        # Calculate sleep duration (matches your Python logic)
        fmt = "%H:%M"
        start = datetime.strptime(sleep_start, fmt)
        end = datetime.strptime(sleep_end, fmt)
        if end < start:  # overnight case
            end += timedelta(days=1)
        duration_hours = (end - start).seconds / 3600

        entry = {
            "date": date,
            "sleep": {
                "start_time": sleep_start,
                "end_time": sleep_end,
                "duration_hours": round(duration_hours, 2),
                "quality_score": quality
            },
            "mood": {
                "mood_score": mood_score,
                "stress_score": stress_score,
                "journal_entry": journal,
                "voice_note_path": voice_note
            },
            "symptoms": {
                "gi_flare": gi_flare,
                "skin_flare": skin_flare,
                "migraine": migraine
            }
        }
        
        # Append to local file
        with open(self.data_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        # Also send to API
        try:
            response = requests.post(f"{self.api_base_url}/health/entries", json=entry)
            if response.status_code == 201:
                print(f"✅ Entry for {date} synced to API")
            else:
                print(f"⚠️  Failed to sync entry for {date}: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  API sync failed for {date}: {e}")
        
        return entry
    
    def load_dataset(self):
        """Load dataset (matches your Python function)"""
        if os.path.getsize(self.data_file) == 0:
            return pd.DataFrame()
        df = pd.read_json(self.data_file, lines=True)
        return pd.json_normalize(df.to_dict(orient="records"))
    
    def sync_from_api(self):
        """Sync data from the API to local file"""
        try:
            response = requests.get(f"{self.api_base_url}/health/entries")
            if response.status_code == 200:
                data = response.json()
                entries = data.get('data', [])
                
                # Write to local file
                with open(self.data_file, "w") as f:
                    for entry in entries:
                        f.write(json.dumps(entry) + "\n")
                
                print(f"✅ Synced {len(entries)} entries from API")
                return entries
            else:
                print(f"❌ Failed to sync from API: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"❌ API sync failed: {e}")
            return []
    
    def sync_to_api(self):
        """Sync local data to the API"""
        try:
            # Read local data
            if not os.path.exists(self.data_file) or os.path.getsize(self.data_file) == 0:
                print("📝 No local data to sync")
                return
            
            with open(self.data_file, "r") as f:
                lines = f.readlines()
            
            synced_count = 0
            for line in lines:
                if line.strip():
                    entry = json.loads(line.strip())
                    try:
                        response = requests.post(f"{self.api_base_url}/health/entries", json=entry)
                        if response.status_code == 201:
                            synced_count += 1
                        else:
                            print(f"⚠️  Failed to sync entry for {entry['date']}: {response.status_code}")
                    except requests.exceptions.RequestException as e:
                        print(f"⚠️  Failed to sync entry for {entry['date']}: {e}")
            
            print(f"✅ Synced {synced_count} entries to API")
        except Exception as e:
            print(f"❌ Sync to API failed: {e}")
    
    def get_correlation_data(self):
        """Get correlation data for analysis"""
        try:
            response = requests.get(f"{self.api_base_url}/health/correlation")
            if response.status_code == 200:
                data = response.json()
                return data.get('data', {})
            else:
                print(f"❌ Failed to get correlation data: {response.status_code}")
                return {}
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get correlation data: {e}")
            return {}
    
    def analyze_correlations(self):
        """Analyze correlations (matches your Python analysis)"""
        df_flat = self.load_dataset()
        
        if df_flat.empty:
            print("📝 No data available for analysis")
            return
        
        print("📊 Sample Data:")
        print(df_flat.head())
        
        # Simple correlation check (matches your Python code)
        correlations = df_flat[[
            "sleep.duration_hours",
            "sleep.quality_score",
            "mood.mood_score",
            "mood.stress_score",
            "symptoms.gi_flare",
            "symptoms.skin_flare",
            "symptoms.migraine"
        ]].corr()
        
        print("\n📈 Correlation Matrix:")
        print(correlations)
        
        return correlations
    
    def export_data(self, filename="health_data_export.json"):
        """Export data for backup or analysis"""
        try:
            response = requests.get(f"{self.api_base_url}/health/export")
            if response.status_code == 200:
                with open(filename, "w") as f:
                    f.write(response.text)
                print(f"✅ Data exported to {filename}")
            else:
                print(f"❌ Export failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Export failed: {e}")
    
    def import_data(self, filename):
        """Import data from file"""
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            
            response = requests.post(f"{self.api_base_url}/health/import", json={"data": data})
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Imported {result.get('message', 'data')}")
            else:
                print(f"❌ Import failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Import failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Sleep & Stress + Python Integration")
    parser.add_argument("--api-url", default=API_BASE_URL, help="API base URL")
    parser.add_argument("--action", choices=[
        "log", "sync-from-api", "sync-to-api", "analyze", "export", "import"
    ], required=True, help="Action to perform")
    parser.add_argument("--date", help="Date for log entry (YYYY-MM-DD)")
    parser.add_argument("--sleep-start", help="Sleep start time (HH:MM)")
    parser.add_argument("--sleep-end", help="Sleep end time (HH:MM)")
    parser.add_argument("--quality", type=int, help="Sleep quality score (1-10)")
    parser.add_argument("--mood", type=int, help="Mood score (1-10)")
    parser.add_argument("--stress", type=int, help="Stress score (1-10)")
    parser.add_argument("--journal", help="Journal entry")
    parser.add_argument("--gi-flare", type=int, default=0, help="GI flare score (0-10)")
    parser.add_argument("--skin-flare", type=int, default=0, help="Skin flare score (0-10)")
    parser.add_argument("--migraine", type=int, default=0, help="Migraine score (0-10)")
    parser.add_argument("--file", help="File for import/export")
    
    args = parser.parse_args()
    
    integration = SleepStressPlusIntegration(args.api_url)
    integration.init_dataset()
    
    if args.action == "log":
        if not all([args.date, args.sleep_start, args.sleep_end, args.quality, args.mood, args.stress]):
            print("❌ Missing required arguments for log entry")
            return
        
        entry = integration.log_entry(
            args.date, args.sleep_start, args.sleep_end, args.quality,
            args.mood, args.stress, args.journal or "", None,
            args.gi_flare, args.skin_flare, args.migraine
        )
        print(f"✅ Logged entry for {args.date}")
    
    elif args.action == "sync-from-api":
        integration.sync_from_api()
    
    elif args.action == "sync-to-api":
        integration.sync_to_api()
    
    elif args.action == "analyze":
        integration.analyze_correlations()
    
    elif args.action == "export":
        filename = args.file or "health_data_export.json"
        integration.export_data(filename)
    
    elif args.action == "import":
        if not args.file:
            print("❌ File required for import")
            return
        integration.import_data(args.file)

if __name__ == "__main__":
    main()
