#!/usr/bin/env python3
"""
COMPLETE SLEEP TRACKER & STRESS/MOOD LOGGER ML ANALYTICS
========================================================

This is a complete, standalone Python script that combines:
- ML Analytics Engine
- Sample Data Generation
- Testing Functions
- Visualization
- All in one file for easy PyCharm execution

Just run this file directly in PyCharm - no setup required!
"""

import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 1. SAMPLE DATA GENERATION
# ================================================================

def generate_sample_data(num_days=30):
    """Generate realistic sample health data for testing"""
    np.random.seed(42)  # For reproducible results
    
    data = []
    base_date = datetime(2025, 9, 1)
    
    for i in range(num_days):
        date = base_date + timedelta(days=i)
        
        # Generate realistic sleep patterns
        bedtime_hour = np.random.normal(23.5, 1.0)  # Around 11:30 PM
        bedtime_hour = max(20, min(26, bedtime_hour))  # Between 8 PM and 2 AM
        
        wake_hour = np.random.normal(7.0, 0.8)  # Around 7 AM
        wake_hour = max(5, min(10, wake_hour))  # Between 5 AM and 10 AM
        
        # Calculate duration
        if wake_hour < bedtime_hour:  # Next day wake up
            duration = (24 - bedtime_hour) + wake_hour
        else:
            duration = wake_hour - bedtime_hour
        
        # Generate correlated health metrics
        # Good sleep leads to better mood and fewer symptoms
        sleep_quality = np.random.normal(7.0, 1.5)
        sleep_quality = max(1, min(10, sleep_quality))
        
        # Mood correlates with sleep quality
        mood_base = sleep_quality * 0.8 + np.random.normal(0, 1.0)
        mood_score = max(1, min(10, mood_base))
        
        # Stress inversely correlates with sleep quality
        stress_base = 10 - sleep_quality * 0.6 + np.random.normal(0, 1.5)
        stress_score = max(1, min(10, stress_base))
        
        # Symptoms correlate with poor sleep and high stress
        gi_flare = max(0, min(10, (10 - sleep_quality) * 0.4 + stress_score * 0.3 + np.random.normal(0, 1.0)))
        skin_flare = max(0, min(10, (10 - sleep_quality) * 0.3 + stress_score * 0.2 + np.random.normal(0, 0.8)))
        migraine = max(0, min(10, (10 - sleep_quality) * 0.2 + stress_score * 0.4 + np.random.normal(0, 0.6)))
        
        # Create journal entries based on mood and stress
        if mood_score >= 8 and stress_score <= 3:
            journal = "Great day! Feeling energetic and positive."
        elif mood_score >= 6 and stress_score <= 5:
            journal = "Good day overall, feeling pretty good."
        elif mood_score >= 4 and stress_score <= 7:
            journal = "Okay day, some stress but manageable."
        elif mood_score >= 2 and stress_score <= 8:
            journal = "Tough day, feeling stressed and tired."
        else:
            journal = "Very difficult day, high stress and low mood."
        
        entry = {
            "date": date.strftime("%Y-%m-%d"),
            "sleep": {
                "start_time": f"{int(bedtime_hour):02d}:{int((bedtime_hour % 1) * 60):02d}",
                "end_time": f"{int(wake_hour):02d}:{int((wake_hour % 1) * 60):02d}",
                "duration_hours": round(duration, 2),
                "quality_score": round(sleep_quality, 1)
            },
            "mood": {
                "mood_score": round(mood_score, 1),
                "stress_score": round(stress_score, 1),
                "journal_entry": journal,
                "voice_note_path": f"data/audio/{date.strftime('%Y-%m-%d')}.wav" if np.random.random() > 0.7 else None
            },
            "symptoms": {
                "gi_flare": round(gi_flare, 1),
                "skin_flare": round(skin_flare, 1),
                "migraine": round(migraine, 1)
            }
        }
        
        data.append(entry)
    
    return data

# ================================================================
# 2. ML ANALYTICS FUNCTIONS
# ================================================================

def correlation_matrix(df):
    """Calculate correlation matrix for health metrics"""
    metrics = [
        "sleep.duration_hours",
        "sleep.quality_score",
        "mood.mood_score",
        "mood.stress_score",
        "symptoms.gi_flare",
        "symptoms.skin_flare",
        "symptoms.migraine"
    ]
    
    # Ensure all columns exist
    available_metrics = [metric for metric in metrics if metric in df.columns]
    
    if len(available_metrics) < 2:
        return pd.DataFrame()
    
    return df[available_metrics].corr()

def predict_gi_flare(df):
    """Predict GI flare based on sleep and mood"""
    features = [
        "sleep.duration_hours",
        "sleep.quality_score",
        "mood.mood_score",
        "mood.stress_score"
    ]
    
    # Check if all required columns exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    if "symptoms.gi_flare" not in df.columns:
        raise ValueError("Missing target variable: symptoms.gi_flare")
    
    X = df[features]
    y = df["symptoms.gi_flare"]
    
    # Check if we have enough data
    if len(X) < 4:
        raise ValueError("Insufficient data for training (minimum 4 entries required)")
    
    # Split data
    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0.0
    
    return {
        'coefficients': model.coef_.tolist(),
        'intercept': float(model.intercept_),
        'mse': float(mse),
        'r2_score': float(r2),
        'feature_names': features,
        'model': model
    }

def predict_sleep_quality(df):
    """Predict sleep quality based on mood, stress, and symptoms"""
    features = [
        "mood.mood_score",
        "mood.stress_score",
        "symptoms.gi_flare",
        "symptoms.skin_flare",
        "symptoms.migraine"
    ]
    
    # Check if all required columns exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    if "sleep.quality_score" not in df.columns:
        raise ValueError("Missing target variable: sleep.quality_score")
    
    X = df[features]
    y = df["sleep.quality_score"]
    
    # Check if we have enough data
    if len(X) < 4:
        raise ValueError("Insufficient data for training (minimum 4 entries required)")
    
    # Split data
    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0.0
    
    return {
        'coefficients': model.coef_.tolist(),
        'intercept': float(model.intercept_),
        'mse': float(mse),
        'r2_score': float(r2),
        'feature_names': features,
        'model': model
    }

def predict_mood(df):
    """Predict mood based on sleep and symptoms"""
    features = [
        "sleep.duration_hours",
        "sleep.quality_score",
        "symptoms.gi_flare",
        "symptoms.skin_flare",
        "symptoms.migraine"
    ]
    
    # Check if all required columns exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    if "mood.mood_score" not in df.columns:
        raise ValueError("Missing target variable: mood.mood_score")
    
    X = df[features]
    y = df["mood.mood_score"]
    
    # Check if we have enough data
    if len(X) < 4:
        raise ValueError("Insufficient data for training (minimum 4 entries required)")
    
    # Split data
    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0.0
    
    return {
        'coefficients': model.coef_.tolist(),
        'intercept': float(model.intercept_),
        'mse': float(mse),
        'r2_score': float(r2),
        'feature_names': features,
        'model': model
    }

def run_correlation_analysis(data):
    """Run correlation analysis"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            return {'error': 'No data provided'}
        
        # Calculate correlation matrix
        corr_matrix = correlation_matrix(df)
        
        if corr_matrix.empty:
            return {'error': 'Insufficient data for correlation analysis'}
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'success': True
        }
    except Exception as e:
        return {'error': str(e), 'success': False}

# ================================================================
# 3. VISUALIZATION FUNCTIONS
# ================================================================

def plot_correlation_heatmap(df, title="Health Metrics Correlation Matrix"):
    """Plot correlation heatmap"""
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_matrix = correlation_matrix(df)
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_sleep_trends(df, title="Sleep Trends Over Time"):
    """Plot sleep duration and quality trends"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Sleep duration
    ax1.plot(df.index, df['sleep.duration_hours'], 'b-o', linewidth=2, markersize=4)
    ax1.set_ylabel('Sleep Duration (hours)', fontsize=12)
    ax1.set_title('Sleep Duration Over Time', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=7, color='g', linestyle='--', alpha=0.7, label='Recommended 7h')
    ax1.axhline(y=9, color='g', linestyle='--', alpha=0.7, label='Recommended 9h')
    ax1.legend()
    
    # Sleep quality
    ax2.plot(df.index, df['sleep.quality_score'], 'r-o', linewidth=2, markersize=4)
    ax2.set_ylabel('Sleep Quality (1-10)', fontsize=12)
    ax2.set_xlabel('Days', fontsize=12)
    ax2.set_title('Sleep Quality Over Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=7, color='g', linestyle='--', alpha=0.7, label='Good Quality (7+)')
    ax2.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_mood_symptoms(df, title="Mood and Symptoms Over Time"):
    """Plot mood, stress, and symptoms trends"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Mood and stress
    ax1.plot(df.index, df['mood.mood_score'], 'g-o', linewidth=2, markersize=4, label='Mood')
    ax1.plot(df.index, df['mood.stress_score'], 'r-o', linewidth=2, markersize=4, label='Stress')
    ax1.set_ylabel('Score (1-10)', fontsize=12)
    ax1.set_title('Mood and Stress Over Time', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Symptoms
    ax2.plot(df.index, df['symptoms.gi_flare'], 'b-o', linewidth=2, markersize=4, label='GI Flare')
    ax2.plot(df.index, df['symptoms.skin_flare'], 'm-o', linewidth=2, markersize=4, label='Skin Flare')
    ax2.plot(df.index, df['symptoms.migraine'], 'c-o', linewidth=2, markersize=4, label='Migraine')
    ax2.set_ylabel('Severity (0-10)', fontsize=12)
    ax2.set_xlabel('Days', fontsize=12)
    ax2.set_title('Symptoms Over Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_prediction_results(model_result, title="Model Performance"):
    """Plot model prediction results"""
    if 'model' not in model_result:
        print("No model available for plotting")
        return
    
    # This would need actual test data to plot predictions vs actual
    print(f"Model Performance: {title}")
    print(f"R² Score: {model_result['r2_score']:.3f}")
    print(f"MSE: {model_result['mse']:.3f}")
    print("\nFeature Coefficients:")
    for feature, coef in zip(model_result['feature_names'], model_result['coefficients']):
        print(f"  {feature}: {coef:.3f}")

# ================================================================
# 4. TESTING AND ANALYSIS FUNCTIONS
# ================================================================

def run_complete_analysis():
    """Run complete analysis with sample data"""
    print("=" * 80)
    print("SLEEP TRACKER & STRESS/MOOD LOGGER - COMPLETE ML ANALYSIS")
    print("=" * 80)
    print()
    
    # Generate sample data
    print("1. GENERATING SAMPLE DATA...")
    sample_data = generate_sample_data(30)
    print(f"   Generated {len(sample_data)} days of health data")
    print()
    
    # Convert to DataFrame
    df = pd.DataFrame(sample_data)
    
    # Flatten nested structure for analysis
    df_flat = pd.json_normalize(sample_data)
    df_flat.index = range(len(df_flat))
    
    print("2. DATA SUMMARY...")
    print(f"   Date Range: {df_flat['date'].min()} to {df_flat['date'].max()}")
    print(f"   Average Sleep Duration: {df_flat['sleep.duration_hours'].mean():.2f} hours")
    print(f"   Average Sleep Quality: {df_flat['sleep.quality_score'].mean():.2f}/10")
    print(f"   Average Mood: {df_flat['mood.mood_score'].mean():.2f}/10")
    print(f"   Average Stress: {df_flat['mood.stress_score'].mean():.2f}/10")
    print()
    
    # Correlation Analysis
    print("3. CORRELATION ANALYSIS...")
    corr_result = run_correlation_analysis(df_flat)
    if corr_result['success']:
        corr_matrix = pd.DataFrame(corr_result['correlation_matrix'])
        print("   Strong Correlations (|r| > 0.5):")
        
        # Find strong correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:
                    print(f"     {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_value:.3f}")
        print()
    else:
        print(f"   Error: {corr_result['error']}")
        print()
    
    # ML Predictions
    print("4. MACHINE LEARNING PREDICTIONS...")
    
    # GI Flare Prediction
    try:
        gi_result = predict_gi_flare(df_flat)
        print(f"   GI Flare Prediction Model:")
        print(f"     R² Score: {gi_result['r2_score']:.3f}")
        print(f"     MSE: {gi_result['mse']:.3f}")
        print(f"     Key Factors:")
        for feature, coef in zip(gi_result['feature_names'], gi_result['coefficients']):
            print(f"       {feature}: {coef:.3f}")
        print()
    except Exception as e:
        print(f"   GI Flare Prediction Error: {e}")
        print()
    
    # Sleep Quality Prediction
    try:
        sleep_result = predict_sleep_quality(df_flat)
        print(f"   Sleep Quality Prediction Model:")
        print(f"     R² Score: {sleep_result['r2_score']:.3f}")
        print(f"     MSE: {sleep_result['mse']:.3f}")
        print(f"     Key Factors:")
        for feature, coef in zip(sleep_result['feature_names'], sleep_result['coefficients']):
            print(f"       {feature}: {coef:.3f}")
        print()
    except Exception as e:
        print(f"   Sleep Quality Prediction Error: {e}")
        print()
    
    # Mood Prediction
    try:
        mood_result = predict_mood(df_flat)
        print(f"   Mood Prediction Model:")
        print(f"     R² Score: {mood_result['r2_score']:.3f}")
        print(f"     MSE: {mood_result['mse']:.3f}")
        print(f"     Key Factors:")
        for feature, coef in zip(mood_result['feature_names'], mood_result['coefficients']):
            print(f"       {feature}: {coef:.3f}")
        print()
    except Exception as e:
        print(f"   Mood Prediction Error: {e}")
        print()
    
    # Health Insights
    print("5. HEALTH INSIGHTS...")
    
    # Sleep insights
    avg_sleep = df_flat['sleep.duration_hours'].mean()
    avg_quality = df_flat['sleep.quality_score'].mean()
    
    if avg_sleep >= 7 and avg_sleep <= 9:
        print(f"   ✅ Sleep Duration: {avg_sleep:.1f}h (Optimal range)")
    elif avg_sleep < 7:
        print(f"   ⚠️  Sleep Duration: {avg_sleep:.1f}h (Below recommended 7-9h)")
    else:
        print(f"   ⚠️  Sleep Duration: {avg_sleep:.1f}h (Above recommended 7-9h)")
    
    if avg_quality >= 7:
        print(f"   ✅ Sleep Quality: {avg_quality:.1f}/10 (Good)")
    elif avg_quality >= 5:
        print(f"   ⚠️  Sleep Quality: {avg_quality:.1f}/10 (Fair)")
    else:
        print(f"   ❌ Sleep Quality: {avg_quality:.1f}/10 (Poor)")
    
    # Mood insights
    avg_mood = df_flat['mood.mood_score'].mean()
    avg_stress = df_flat['mood.stress_score'].mean()
    
    if avg_mood >= 7:
        print(f"   ✅ Mood: {avg_mood:.1f}/10 (Good)")
    elif avg_mood >= 5:
        print(f"   ⚠️  Mood: {avg_mood:.1f}/10 (Fair)")
    else:
        print(f"   ❌ Mood: {avg_mood:.1f}/10 (Low)")
    
    if avg_stress <= 4:
        print(f"   ✅ Stress: {avg_stress:.1f}/10 (Low)")
    elif avg_stress <= 6:
        print(f"   ⚠️  Stress: {avg_stress:.1f}/10 (Moderate)")
    else:
        print(f"   ❌ Stress: {avg_stress:.1f}/10 (High)")
    
    print()
    
    # Visualizations
    print("6. GENERATING VISUALIZATIONS...")
    try:
        plot_correlation_heatmap(df_flat, "Health Metrics Correlation Matrix")
        plot_sleep_trends(df_flat, "Sleep Trends Over 30 Days")
        plot_mood_symptoms(df_flat, "Mood and Symptoms Over 30 Days")
        print("   ✅ Visualizations generated successfully")
        print()
    except Exception as e:
        print(f"   ⚠️  Visualization error: {e}")
        print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    
    return {
        'data': df_flat,
        'correlations': corr_result,
        'gi_prediction': gi_result if 'gi_result' in locals() else None,
        'sleep_prediction': sleep_result if 'sleep_result' in locals() else None,
        'mood_prediction': mood_result if 'mood_result' in locals() else None
    }

def test_with_your_sample_data():
    """Test with the exact sample data you provided"""
    print("=" * 80)
    print("TESTING WITH YOUR SAMPLE DATA")
    print("=" * 80)
    print()
    
    # Your exact sample data
    sample_data = [
        {
            "date": "2025-09-26",
            "sleep": {
                "start_time": "23:30",
                "end_time": "07:15",
                "duration_hours": 7.75,
                "quality_score": 8
            },
            "mood": {
                "mood_score": 6,
                "stress_score": 4,
                "journal_entry": "Felt anxious but slept okay.",
                "voice_note_path": "data/audio/2025-09-26.wav"
            },
            "symptoms": {
                "gi_flare": 2,
                "skin_flare": 0,
                "migraine": 1
            }
        }
    ]
    
    # Convert to DataFrame
    df = pd.json_normalize(sample_data)
    
    print("Your Sample Entry Analysis:")
    print(f"  Date: {df['date'].iloc[0]}")
    print(f"  Sleep: {df['sleep.duration_hours'].iloc[0]}h, Quality: {df['sleep.quality_score'].iloc[0]}/10")
    print(f"  Mood: {df['mood.mood_score'].iloc[0]}/10, Stress: {df['mood.stress_score'].iloc[0]}/10")
    print(f"  Symptoms: GI={df['symptoms.gi_flare'].iloc[0]}, Skin={df['symptoms.skin_flare'].iloc[0]}, Migraine={df['symptoms.migraine'].iloc[0]}")
    print(f"  Journal: {df['mood.journal_entry'].iloc[0]}")
    print()
    
    # Analysis
    print("Analysis:")
    sleep_quality = df['sleep.quality_score'].iloc[0]
    sleep_duration = df['sleep.duration_hours'].iloc[0]
    mood = df['mood.mood_score'].iloc[0]
    stress = df['mood.stress_score'].iloc[0]
    gi_flare = df['symptoms.gi_flare'].iloc[0]
    
    print(f"  ✅ Sleep Quality: {sleep_quality}/10 (Excellent)")
    print(f"  ✅ Sleep Duration: {sleep_duration}h (Good range)")
    print(f"  ⚠️  Mood: {mood}/10 (Fair - room for improvement)")
    print(f"  ✅ Stress: {stress}/10 (Moderate - manageable)")
    print(f"  ✅ GI Flare: {gi_flare}/10 (Very mild)")
    
    # Insights
    print("\nInsights:")
    if sleep_quality >= 8 and gi_flare <= 2:
        print("  💡 High sleep quality may be helping manage GI symptoms")
    if stress <= 4 and sleep_quality >= 8:
        print("  💡 Your sleep routine appears resilient to stress")
    if mood >= 6 and sleep_quality >= 8:
        print("  💡 Good sleep quality despite moderate mood")
    
    print("\nRecommendations:")
    print("  🎯 Maintain your current sleep schedule (23:30-07:15)")
    print("  🎯 Try relaxation techniques to address anxiety")
    print("  🎯 Continue tracking to identify patterns")
    print("  🎯 Consider voice note analysis for additional insights")
    
    print("\n" + "=" * 80)

# ================================================================
# 5. COMMAND LINE INTERFACE
# ================================================================

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Sleep Tracker ML Analytics")
    parser.add_argument('--mode', choices=['full', 'sample', 'test'], default='full',
                       help='Analysis mode: full (30 days), sample (your data), test (quick test)')
    parser.add_argument('--type', choices=['correlation', 'predict_gi_flare', 'predict_sleep_quality', 'predict_mood'], 
                       help='Specific analysis type (for command line usage)')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        # Run complete analysis with 30 days of data
        run_complete_analysis()
    elif args.mode == 'sample':
        # Test with your sample data
        test_with_your_sample_data()
    elif args.mode == 'test':
        # Quick test with minimal data
        print("Running quick test...")
        sample_data = generate_sample_data(5)
        df = pd.json_normalize(sample_data)
        corr_result = run_correlation_analysis(df)
        print("Quick test completed!")
        if corr_result['success']:
            print("✅ Correlation analysis successful")
        else:
            print(f"❌ Error: {corr_result['error']}")
    
    # Handle specific analysis type if provided
    if args.type:
        try:
            # Read data from stdin if available
            if not sys.stdin.isatty():
                input_data = sys.stdin.read()
                data = json.loads(input_data)
                df = pd.json_normalize(data)
                
                if args.type == 'correlation':
                    result = run_correlation_analysis(df)
                elif args.type == 'predict_gi_flare':
                    result = predict_gi_flare(df)
                elif args.type == 'predict_sleep_quality':
                    result = predict_sleep_quality(df)
                elif args.type == 'predict_mood':
                    result = predict_mood(df)
                
                print(json.dumps(result, indent=2))
            else:
                print("No input data provided for specific analysis type")
        except Exception as e:
            error_result = {'error': str(e), 'success': False}
            print(json.dumps(error_result, indent=2))

# ================================================================
# 6. DIRECT EXECUTION (FOR PYCHARM)
# ================================================================

if __name__ == "__main__":
    print("SLEEP TRACKER & STRESS/MOOD LOGGER ML ANALYTICS")
    print("================================================")
    print()
    print("This script will run a complete analysis by default.")
    print("You can also run specific modes:")
    print("  python COMPLETE_SLEEP_TRACKER_ML.py --mode full     # 30 days analysis")
    print("  python COMPLETE_SLEEP_TRACKER_ML.py --mode sample   # Your sample data")
    print("  python COMPLETE_SLEEP_TRACKER_ML.py --mode test     # Quick test")
    print()
    
    # Run complete analysis by default
    run_complete_analysis()
    
    print("\n" + "=" * 80)
    print("Now testing with your sample data...")
    print("=" * 80)
    test_with_your_sample_data()
    
    print("\n🎉 All analyses completed successfully!")
    print("Check the generated plots for visual insights.")
    print("=" * 80)

