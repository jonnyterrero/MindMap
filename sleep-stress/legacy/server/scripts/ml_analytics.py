#!/usr/bin/env python3
"""
ML Analytics Script for Sleep & Stress + App
This script runs the ML models and correlation analysis
"""

import json
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import argparse

def correlation_matrix(df):
    """Calculate correlation matrix (matches your Python function)"""
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
    """Predict GI flare (matches your Python function)"""
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
        # Use all data for training if we have less than 10 entries
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
        'feature_names': features
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
        # Use all data for training if we have less than 10 entries
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
        'feature_names': features
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
        # Use all data for training if we have less than 10 entries
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
        'feature_names': features
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

def run_gi_flare_prediction(data):
    """Run GI flare prediction"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            return {'error': 'No data provided'}
        
        # Run prediction
        result = predict_gi_flare(df)
        result['success'] = True
        
        return result
    except Exception as e:
        return {'error': str(e), 'success': False}

def run_sleep_quality_prediction(data):
    """Run sleep quality prediction"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            return {'error': 'No data provided'}
        
        # Run prediction
        result = predict_sleep_quality(df)
        result['success'] = True
        
        return result
    except Exception as e:
        return {'error': str(e), 'success': False}

def run_mood_prediction(data):
    """Run mood prediction"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            return {'error': 'No data provided'}
        
        # Run prediction
        result = predict_mood(df)
        result['success'] = True
        
        return result
    except Exception as e:
        return {'error': str(e), 'success': False}

def main():
    parser = argparse.ArgumentParser(description="ML Analytics for Sleep & Stress +")
    parser.add_argument('--type', choices=['correlation', 'predict_gi_flare', 'predict_sleep_quality', 'predict_mood'], required=True,
                       help='Type of analysis to run')
    
    args = parser.parse_args()
    
    try:
        # Read data from stdin
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        
        if args.type == 'correlation':
            result = run_correlation_analysis(data)
        elif args.type == 'predict_gi_flare':
            result = run_gi_flare_prediction(data)
        elif args.type == 'predict_sleep_quality':
            result = run_sleep_quality_prediction(data)
        elif args.type == 'predict_mood':
            result = run_mood_prediction(data)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {'error': str(e), 'success': False}
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()
