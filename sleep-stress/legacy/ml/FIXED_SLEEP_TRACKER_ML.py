#!/usr/bin/env python3
"""
FIXED SLEEP TRACKER & STRESS/MOOD LOGGER ML ANALYTICS
=====================================================

This fixes all the issues in your original code:
- SettingWithCopyWarning resolved
- Proper data generation for training
- Better feature engineering
- Improved error handling
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# ================================
# CONFIG
# ================================
class Config:
    DATA_FILE = "health_logs.jsonl"
    MODEL_DIR = "models"
    SEQ_LEN = 7  # Reduced from 14 for smaller datasets
    PRED_HORIZON = 1
    BATCH_SIZE = 8  # Reduced for smaller datasets
    HIDDEN_SIZE = 32  # Reduced for smaller datasets
    NUM_LAYERS = 2
    DROPOUT = 0.2
    LR = 1e-3
    EPOCHS = 20  # Reduced for faster training
    RNN_TYPE = "gru"  # or "lstm"

    # alert thresholds
    LOW_SLEEP_HOURS = 6.0
    HIGH_STRESS = 7.0
    HIGH_RISK_THRESHOLD = 6.5


# ================================
# DATA HANDLING
# ================================
def init_dataset():
    if not os.path.exists(Config.DATA_FILE):
        with open(Config.DATA_FILE, "w") as f:
            pass

def log_entry(date, sleep_start, sleep_end, quality, mood_score, stress_score,
              journal="", voice_note=None, gi_flare=0, skin_flare=0, migraine=0):
    fmt = "%H:%M"
    start = datetime.strptime(sleep_start, fmt)
    end = datetime.strptime(sleep_end, fmt)
    if end < start:
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
    with open(Config.DATA_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def load_dataset():
    if not os.path.exists(Config.DATA_FILE) or os.path.getsize(Config.DATA_FILE) == 0:
        return pd.DataFrame()
    df = pd.read_json(Config.DATA_FILE, lines=True)
    return pd.json_normalize(df.to_dict(orient="records"))

def generate_sample_data(num_days=30):
    """Generate realistic sample data for training"""
    print(f"Generating {num_days} days of sample data...")
    
    # Clear existing data
    if os.path.exists(Config.DATA_FILE):
        os.remove(Config.DATA_FILE)
    init_dataset()
    
    np.random.seed(42)
    base_date = datetime(2025, 9, 1)
    
    for i in range(num_days):
        date = base_date + timedelta(days=i)
        
        # Generate realistic sleep patterns
        bedtime_hour = np.random.normal(23.5, 1.0)
        bedtime_hour = max(20, min(23.99, bedtime_hour))  # Cap at 23.99 to avoid 24:xx
        
        wake_hour = np.random.normal(7.0, 0.8)
        wake_hour = max(5, min(10, wake_hour))
        
        # Calculate duration
        if wake_hour < bedtime_hour:
            duration = (24 - bedtime_hour) + wake_hour
        else:
            duration = wake_hour - bedtime_hour
        
        # Ensure valid time format
        bedtime_hour = int(bedtime_hour)
        bedtime_min = int((bedtime_hour % 1) * 60)
        wake_hour = int(wake_hour)
        wake_min = int((wake_hour % 1) * 60)
        
        # Format times properly
        bedtime_str = f"{bedtime_hour:02d}:{bedtime_min:02d}"
        wake_str = f"{wake_hour:02d}:{wake_min:02d}"
        
        # Generate correlated health metrics
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
        
        # Create journal entries
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
        
        # Log entry
        log_entry(
            date.strftime("%Y-%m-%d"),
            bedtime_str,
            wake_str,
            round(sleep_quality, 1),
            round(mood_score, 1),
            round(stress_score, 1),
            journal,
            f"data/audio/{date.strftime('%Y-%m-%d')}.wav" if np.random.random() > 0.7 else None,
            round(gi_flare, 1),
            round(skin_flare, 1),
            round(migraine, 1)
        )

# ================================
# FEATURE ENGINEERING (FIXED)
# ================================
def build_features():
    df = load_dataset()
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # Create a copy to avoid SettingWithCopyWarning
    num_df = df.select_dtypes("number").copy()
    
    # Add rolling averages
    for col in num_df.columns:
        for w in [3, 7]:
            # Use .loc to avoid SettingWithCopyWarning
            num_df.loc[:, f"{col}_ma{w}"] = num_df[col].rolling(w, min_periods=1).mean()
    
    # Add day of week and time features
    num_df.loc[:, "day_of_week"] = df.index.dayofweek
    num_df.loc[:, "is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    
    # Add lag features
    for col in ["sleep.duration_hours", "sleep.quality_score", "mood.mood_score", "mood.stress_score"]:
        if col in num_df.columns:
            num_df.loc[:, f"{col}_lag1"] = num_df[col].shift(1)
            num_df.loc[:, f"{col}_lag2"] = num_df[col].shift(2)

    return num_df


# ================================
# DATASET
# ================================
class SeqDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len=7, horizon=1):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.indices = self._build_indices(len(df))

    def _build_indices(self, n):
        idx = []
        L, H = self.seq_len, self.horizon
        for t in range(n - L - H + 1):
            idx.append((t, t+L, t+L+H-1))
        return idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s, e, t = self.indices[i]
        x_seq = self.X[s:e]
        y_next = self.y[t]
        return torch.from_numpy(x_seq), torch.tensor([y_next], dtype=torch.float32)


# ================================
# MODEL
# ================================
class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2, rnn_type="gru"):
        super().__init__()
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                              batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)


# ================================
# TRAINING (IMPROVED)
# ================================
def train_model(df, feature_cols, target_col, model_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {target_col} model on {DEVICE}...")

    # Handle missing values
    df_clean = df.dropna()
    if len(df_clean) < Config.SEQ_LEN + 5:  # Need at least seq_len + 5 for training
        print(f"[WARN] Not enough data to train {target_col} (need at least {Config.SEQ_LEN + 5}, got {len(df_clean)})")
        return False

    scaler = StandardScaler()
    df_scaled = df_clean.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_clean[feature_cols])

    dataset = SeqDataset(df_scaled, feature_cols, target_col,
                         seq_len=Config.SEQ_LEN, horizon=Config.PRED_HORIZON)
    
    if len(dataset) < 5:
        print(f"[WARN] Not enough sequences to train {target_col}")
        return False

    # Split data
    val_size = max(1, min(5, int(0.3 * len(dataset))))  # Cap validation size
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=min(Config.BATCH_SIZE, len(train_ds)), shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=min(Config.BATCH_SIZE, len(val_ds)))

    model = RNNRegressor(len(feature_cols), Config.HIDDEN_SIZE,
                         Config.NUM_LAYERS, Config.DROPOUT, Config.RNN_TYPE).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=Config.LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    for epoch in range(1, Config.EPOCHS+1):
        # Training
        model.train()
        tr_loss = 0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * Xb.size(0)
        tr_loss /= len(train_dl.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_loss += loss_fn(model(Xb), yb).item() * Xb.size(0)
        val_loss /= len(val_dl.dataset)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "scaler_mean": scaler.mean_,
                        "scaler_scale": scaler.scale_,
                        "feature_cols": feature_cols}, model_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{target_col}] Epoch {epoch} | Train {tr_loss:.4f} | Val {val_loss:.4f}")

    print(f"✅ Saved {target_col} model to {model_path}")
    return True


# ================================
# INFERENCE + ALERTS
# ================================
def predict_next(df, target_col, model_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(model_path):
        return None

    try:
        ckpt = torch.load(model_path, map_location=DEVICE)
        feature_cols = ckpt["feature_cols"]
        scaler = StandardScaler()
        scaler.mean_ = ckpt["scaler_mean"]
        scaler.scale_ = ckpt["scaler_scale"]

        df_ = df.copy()
        df_[feature_cols] = scaler.transform(df_[feature_cols])
        seq = df_.tail(Config.SEQ_LEN)[feature_cols].values.astype(np.float32)
        if len(seq) < Config.SEQ_LEN:
            return None
        x = torch.tensor(seq).unsqueeze(0).to(DEVICE)

        model = RNNRegressor(len(feature_cols), Config.HIDDEN_SIZE,
                             Config.NUM_LAYERS, Config.DROPOUT, Config.RNN_TYPE).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            return model(x).item()
    except Exception as e:
        print(f"Error predicting {target_col}: {e}")
        return None

def generate_alerts(df, preds):
    alerts = []
    if df.empty: 
        return alerts
    last = df.tail(1)

    # Rule-based alerts
    sleep = float(last["sleep.duration_hours"].iloc[0])
    stress = float(last["mood.stress_score"].iloc[0])
    
    if sleep < Config.LOW_SLEEP_HOURS:
        alerts.append(f"⚠️ Low sleep ({sleep:.1f}h) - aim for 7-9 hours")
    if stress >= Config.HIGH_STRESS:
        alerts.append(f"⚠️ High stress ({stress:.1f}/10) - consider relaxation techniques")
    if sleep < Config.LOW_SLEEP_HOURS and stress >= Config.HIGH_STRESS:
        alerts.append("🚨 Combined risk: low sleep + high stress - prioritize rest")

    # Model-based alerts
    if preds.get("gi_flare") and preds["gi_flare"] >= Config.HIGH_RISK_THRESHOLD:
        alerts.append(f"🔮 Model predicts high GI flare risk tomorrow (~{preds['gi_flare']:.1f}/10)")
    if preds.get("mood") and preds["mood"] <= 4.0:
        alerts.append(f"🔮 Model predicts low mood tomorrow (~{preds['mood']:.1f}/10)")

    return alerts


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    print("=" * 80)
    print("SLEEP TRACKER & STRESS/MOOD LOGGER ML ANALYTICS")
    print("=" * 80)
    print()
    
    init_dataset()

    # Generate sample data for training
    generate_sample_data(30)
    
    # Load and process data
    df = build_features()
    if df.empty:
        print("No data available yet.")
        exit()

    print(f"Loaded {len(df)} days of data")
    print(f"Features: {list(df.columns)}")
    print()

    # Define targets and features
    targets = ["symptoms.gi_flare", "mood.mood_score"]
    feature_cols = [c for c in df.columns if c not in targets]
    
    print(f"Training features: {feature_cols}")
    print(f"Target variables: {targets}")
    print()

    # Train models
    models_trained = 0
    for target in targets:
        if train_model(df, feature_cols, target, os.path.join(Config.MODEL_DIR, f"{target.split('.')[-1]}.pt")):
            models_trained += 1

    print(f"\nTrained {models_trained}/{len(targets)} models")
    print()

    # Predict next day
    preds = {
        "gi_flare": predict_next(df, "symptoms.gi_flare", os.path.join(Config.MODEL_DIR, "gi_flare.pt")),
        "mood": predict_next(df, "mood.mood_score", os.path.join(Config.MODEL_DIR, "mood_score.pt"))
    }
    
    print("🔮 Predictions for tomorrow:")
    for key, value in preds.items():
        if value is not None:
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: No prediction available")
    print()

    # Generate alerts
    alerts = generate_alerts(df, preds)
    if alerts:
        print("🚨 Health Alerts:")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("✅ No health alerts - looking good!")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
