#!/usr/bin/env python3
"""
ADVANCED SLEEP TRACKER & STRESS/MOOD LOGGER ML ANALYTICS
========================================================

Enhanced features:
- Multi-task learning (regression + classification)
- Transformer encoder for longer contexts
- Experiment tracking (Weights & Biases)
- Data quality checks and outlier clipping
- SHAP explainability for alerts
- Advanced model architectures
- Personalized insights and recommendations
- Coping strategies and mood boosters
- Gamification and progress tracking
- Smart notifications and reminders
- Voice transcription simulation
- Trend analysis and correlation heatmaps
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import random

# Optional imports for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Install with: pip install wandb")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Install with: pip install mlflow")

# ================================
# CONFIG
# ================================
class Config:
    DATA_FILE = "health_logs.jsonl"
    MODEL_DIR = "models"
    EXPERIMENT_DIR = "experiments"
    
    # Model architecture
    SEQ_LEN = 28  # Longer context for transformer
    PRED_HORIZON = 1
    BATCH_SIZE = 16
    HIDDEN_SIZE = 128
    NUM_LAYERS = 4
    NUM_HEADS = 8  # For transformer
    DROPOUT = 0.1
    LR = 1e-4
    EPOCHS = 50
    MODEL_TYPE = "transformer"  # "rnn", "transformer", "multitask"
    
    # Multi-task learning
    FLARE_THRESHOLD = 6.0  # Binary classification threshold
    TASK_WEIGHTS = {"regression": 1.0, "classification": 0.5}
    
    # Data quality
    MIN_DAILY_FIELDS = 5
    OUTLIER_CLIPS = {
        "sleep.duration_hours": (0, 14),
        "sleep.quality_score": (1, 10),
        "mood.mood_score": (1, 10),
        "mood.stress_score": (1, 10),
        "symptoms.gi_flare": (0, 10),
        "symptoms.skin_flare": (0, 10),
        "symptoms.migraine": (0, 10)
    }
    
    # Alert thresholds
    LOW_SLEEP_HOURS = 6.0
    HIGH_STRESS = 7.0
    HIGH_RISK_THRESHOLD = 6.5
    
    # Experiment tracking
    USE_WANDB = True
    USE_MLFLOW = True
    EXPERIMENT_NAME = "sleep_tracker_advanced"


# ================================
# DATA QUALITY & VALIDATION
# ================================
class DataQualityChecker:
    def __init__(self, config):
        self.config = config
        self.quality_report = {}
    
    def check_entry_quality(self, entry):
        """Check if entry has minimum required fields"""
        required_fields = [
            "sleep.duration_hours", "sleep.quality_score",
            "mood.mood_score", "mood.stress_score",
            "symptoms.gi_flare"
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in entry or pd.isna(entry[field]):
                missing_fields.append(field)
        
        return len(missing_fields) == 0, missing_fields
    
    def clip_outliers(self, df):
        """Clip outliers based on config thresholds"""
        df_clean = df.copy()
        clipped_count = 0
        
        for col, (min_val, max_val) in self.config.OUTLIER_CLIPS.items():
            if col in df_clean.columns:
                before_count = len(df_clean[df_clean[col] < min_val]) + len(df_clean[df_clean[col] > max_val])
                df_clean[col] = df_clean[col].clip(min_val, max_val)
                clipped_count += before_count
        
        self.quality_report["outliers_clipped"] = clipped_count
        return df_clean
    
    def validate_dataset(self, df):
        """Comprehensive dataset validation"""
        report = {
            "total_entries": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "outliers_clipped": 0,
            "quality_score": 0.0
        }
        
        # Check data quality
        quality_scores = []
        for idx, row in df.iterrows():
            is_valid, missing = self.check_entry_quality(row)
            quality_scores.append(1.0 if is_valid else 0.0)
        
        report["quality_score"] = np.mean(quality_scores)
        report["valid_entries"] = sum(quality_scores)
        
        return report


# ================================
# DATA HANDLING (ENHANCED)
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

def generate_sample_data(num_days=60):
    """Generate more realistic sample data for longer contexts"""
    print(f"Generating {num_days} days of sample data...")
    
    if os.path.exists(Config.DATA_FILE):
        os.remove(Config.DATA_FILE)
    init_dataset()
    
    np.random.seed(42)
    base_date = datetime(2025, 8, 1)  # Start earlier for longer context
    
    for i in range(num_days):
        date = base_date + timedelta(days=i)
        
        # Generate realistic sleep patterns with seasonal variation
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 0.1 * np.sin(2 * np.pi * day_of_year / 365)
        
        bedtime_hour = np.random.normal(23.5 + seasonal_factor, 1.0)
        bedtime_hour = max(20, min(23.99, bedtime_hour))
        
        wake_hour = np.random.normal(7.0 - seasonal_factor, 0.8)
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
        
        bedtime_str = f"{bedtime_hour:02d}:{bedtime_min:02d}"
        wake_str = f"{wake_hour:02d}:{wake_min:02d}"
        
        # Generate correlated health metrics with more realistic patterns
        sleep_quality = np.random.normal(7.0 + seasonal_factor, 1.5)
        sleep_quality = max(1, min(10, sleep_quality))
        
        # Add some weekly patterns
        weekday_factor = 0.2 if date.weekday() < 5 else -0.3  # Better sleep on weekends
        sleep_quality += weekday_factor
        
        # Mood correlates with sleep quality
        mood_base = sleep_quality * 0.8 + np.random.normal(0, 1.0)
        mood_score = max(1, min(10, mood_base))
        
        # Stress inversely correlates with sleep quality
        stress_base = 10 - sleep_quality * 0.6 + np.random.normal(0, 1.5)
        stress_score = max(1, min(10, stress_base))
        
        # Symptoms with more complex patterns
        gi_flare = max(0, min(10, (10 - sleep_quality) * 0.4 + stress_score * 0.3 + 
                                  np.random.normal(0, 1.0) + weekday_factor * 0.5))
        skin_flare = max(0, min(10, (10 - sleep_quality) * 0.3 + stress_score * 0.2 + 
                                   np.random.normal(0, 0.8)))
        migraine = max(0, min(10, (10 - sleep_quality) * 0.2 + stress_score * 0.4 + 
                                 np.random.normal(0, 0.6)))
        
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
# FEATURE ENGINEERING (ENHANCED)
# ================================
def build_features():
    df = load_dataset()
    if df.empty:
        return df

    # Data quality check
    quality_checker = DataQualityChecker(Config)
    quality_report = quality_checker.validate_dataset(df)
    print(f"Data Quality Score: {quality_report['quality_score']:.2f}")
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # Create a copy to avoid SettingWithCopyWarning
    num_df = df.select_dtypes("number").copy()
    
    # Clip outliers
    num_df = quality_checker.clip_outliers(num_df)
    
    # Enhanced feature engineering
    for col in num_df.columns:
        # Rolling averages
        for w in [3, 7, 14, 28]:
            num_df.loc[:, f"{col}_ma{w}"] = num_df[col].rolling(w, min_periods=1).mean()
        
        # Rolling standard deviations
        for w in [7, 14]:
            num_df.loc[:, f"{col}_std{w}"] = num_df[col].rolling(w, min_periods=1).std()
        
        # Rolling min/max
        for w in [7, 14]:
            num_df.loc[:, f"{col}_min{w}"] = num_df[col].rolling(w, min_periods=1).min()
            num_df.loc[:, f"{col}_max{w}"] = num_df[col].rolling(w, min_periods=1).max()
    
    # Time-based features
    num_df.loc[:, "day_of_week"] = df.index.dayofweek
    num_df.loc[:, "is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    num_df.loc[:, "month"] = df.index.month
    num_df.loc[:, "day_of_year"] = df.index.dayofyear
    
    # Lag features
    for col in ["sleep.duration_hours", "sleep.quality_score", "mood.mood_score", "mood.stress_score"]:
        if col in num_df.columns:
            for lag in [1, 2, 3, 7]:
                num_df.loc[:, f"{col}_lag{lag}"] = num_df[col].shift(lag)
    
    # Interaction features
    if "sleep.duration_hours" in num_df.columns and "sleep.quality_score" in num_df.columns:
        num_df.loc[:, "sleep_efficiency"] = num_df["sleep.duration_hours"] * num_df["sleep.quality_score"]
    
    if "mood.mood_score" in num_df.columns and "mood.stress_score" in num_df.columns:
        num_df.loc[:, "mood_stress_ratio"] = num_df["mood.mood_score"] / (num_df["mood.stress_score"] + 0.1)

    return num_df


# ================================
# ENHANCED DATASET
# ================================
class AdvancedSeqDataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, seq_len=28, horizon=1, flare_threshold=6.0):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y_reg = df[target_cols["regression"]].values.astype(np.float32)
        self.y_cls = (df[target_cols["classification"]] >= flare_threshold).astype(np.float32)
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
        y_reg_next = self.y_reg[t]
        y_cls_next = self.y_cls[t]
        return (torch.from_numpy(x_seq), 
                torch.tensor([y_reg_next], dtype=torch.float32),
                torch.tensor([y_cls_next], dtype=torch.float32))


# ================================
# TRANSFORMER MODEL
# ================================
class TransformerRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_heads=8, num_layers=4, 
                 dropout=0.1, seq_len=28):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, hidden_size))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, hidden_size]
        
        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(0)
        
        # Transformer encoding
        encoded = self.transformer(x)  # [batch, seq_len, hidden_size]
        
        # Global average pooling
        pooled = encoded.mean(dim=1)  # [batch, hidden_size]
        
        # Multi-task outputs
        reg_output = self.regression_head(pooled)
        cls_output = self.classification_head(pooled)
        
        return reg_output, cls_output


# ================================
# EXPERIMENT TRACKING
# ================================
class ExperimentTracker:
    def __init__(self, config):
        self.config = config
        self.wandb_run = None
        self.mlflow_run = None
        
    def init_tracking(self, model_name="transformer"):
        if self.config.USE_WANDB and WANDB_AVAILABLE:
            # Convert config to regular dict to avoid pickle issues
            config_dict = {
                "SEQ_LEN": self.config.SEQ_LEN,
                "PRED_HORIZON": self.config.PRED_HORIZON,
                "BATCH_SIZE": self.config.BATCH_SIZE,
                "HIDDEN_SIZE": self.config.HIDDEN_SIZE,
                "NUM_LAYERS": self.config.NUM_LAYERS,
                "NUM_HEADS": self.config.NUM_HEADS,
                "DROPOUT": self.config.DROPOUT,
                "LR": self.config.LR,
                "EPOCHS": self.config.EPOCHS,
                "MODEL_TYPE": self.config.MODEL_TYPE,
                "FLARE_THRESHOLD": self.config.FLARE_THRESHOLD,
                "TASK_WEIGHTS": self.config.TASK_WEIGHTS
            }
            
            self.wandb_run = wandb.init(
                project="sleep-tracker",
                name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config_dict
            )
        
        if self.config.USE_MLFLOW and MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.config.EXPERIMENT_NAME)
            self.mlflow_run = mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    def log_metrics(self, metrics, epoch=None):
        if self.wandb_run:
            if epoch is not None:
                metrics["epoch"] = epoch
            wandb.log(metrics)
        
        if self.mlflow_run:
            mlflow.log_metrics(metrics, step=epoch)
    
    def log_model(self, model, model_name):
        if self.mlflow_run:
            mlflow.pytorch.log_model(model, model_name)
    
    def finish(self):
        if self.wandb_run:
            wandb.finish()
        if self.mlflow_run:
            mlflow.end_run()


# ================================
# ENHANCED TRAINING
# ================================
def train_multitask_model(df, feature_cols, target_cols, model_path, tracker=None):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training multi-task model on {DEVICE}...")

    # Handle missing values
    df_clean = df.dropna()
    if len(df_clean) < Config.SEQ_LEN + 10:
        print(f"[WARN] Not enough data to train (need at least {Config.SEQ_LEN + 10}, got {len(df_clean)})")
        return False

    # Use RobustScaler for better outlier handling
    scaler = RobustScaler()
    df_scaled = df_clean.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_clean[feature_cols])

    dataset = AdvancedSeqDataset(df_scaled, feature_cols, target_cols,
                                seq_len=Config.SEQ_LEN, horizon=Config.PRED_HORIZON,
                                flare_threshold=Config.FLARE_THRESHOLD)
    
    if len(dataset) < 10:
        print(f"[WARN] Not enough sequences to train")
        return False

    # Split data
    val_size = max(5, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=Config.BATCH_SIZE)

    # Initialize model
    model = TransformerRegressor(len(feature_cols), Config.HIDDEN_SIZE,
                                Config.NUM_HEADS, Config.NUM_LAYERS, 
                                Config.DROPOUT, Config.SEQ_LEN).to(DEVICE)
    
    optim = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-5)
    reg_loss_fn = nn.MSELoss()
    cls_loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    for epoch in range(1, Config.EPOCHS + 1):
        # Training
        model.train()
        train_reg_loss = 0
        train_cls_loss = 0
        train_total_loss = 0
        
        for Xb, y_reg, y_cls in train_dl:
            Xb, y_reg, y_cls = Xb.to(DEVICE), y_reg.to(DEVICE), y_cls.to(DEVICE)
            
            optim.zero_grad()
            reg_pred, cls_pred = model(Xb)
            
            reg_loss = reg_loss_fn(reg_pred, y_reg)
            cls_loss = cls_loss_fn(cls_pred, y_cls)
            total_loss = (Config.TASK_WEIGHTS["regression"] * reg_loss + 
                         Config.TASK_WEIGHTS["classification"] * cls_loss)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            
            train_reg_loss += reg_loss.item() * Xb.size(0)
            train_cls_loss += cls_loss.item() * Xb.size(0)
            train_total_loss += total_loss.item() * Xb.size(0)
        
        train_reg_loss /= len(train_dl.dataset)
        train_cls_loss /= len(train_dl.dataset)
        train_total_loss /= len(train_dl.dataset)

        # Validation
        model.eval()
        val_reg_loss = 0
        val_cls_loss = 0
        val_total_loss = 0
        val_cls_correct = 0
        val_cls_total = 0
        
        with torch.no_grad():
            for Xb, y_reg, y_cls in val_dl:
                Xb, y_reg, y_cls = Xb.to(DEVICE), y_reg.to(DEVICE), y_cls.to(DEVICE)
                
                reg_pred, cls_pred = model(Xb)
                
                reg_loss = reg_loss_fn(reg_pred, y_reg)
                cls_loss = cls_loss_fn(cls_pred, y_cls)
                total_loss = (Config.TASK_WEIGHTS["regression"] * reg_loss + 
                             Config.TASK_WEIGHTS["classification"] * cls_loss)
                
                val_reg_loss += reg_loss.item() * Xb.size(0)
                val_cls_loss += cls_loss.item() * Xb.size(0)
                val_total_loss += total_loss.item() * Xb.size(0)
                
                # Classification accuracy
                cls_pred_binary = (torch.sigmoid(cls_pred) > 0.5).float()
                val_cls_correct += (cls_pred_binary == y_cls).sum().item()
                val_cls_total += y_cls.size(0)
        
        val_reg_loss /= len(val_dl.dataset)
        val_cls_loss /= len(val_dl.dataset)
        val_total_loss /= len(val_dl.dataset)
        val_cls_acc = val_cls_correct / val_cls_total

        # Log metrics
        metrics = {
            "train_reg_loss": train_reg_loss,
            "train_cls_loss": train_cls_loss,
            "train_total_loss": train_total_loss,
            "val_reg_loss": val_reg_loss,
            "val_cls_loss": val_cls_loss,
            "val_total_loss": val_total_loss,
            "val_cls_accuracy": val_cls_acc
        }
        
        if tracker:
            tracker.log_metrics(metrics, epoch)

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            # Convert config to regular dict to avoid pickle issues
            config_dict = {
                "SEQ_LEN": Config.SEQ_LEN,
                "PRED_HORIZON": Config.PRED_HORIZON,
                "BATCH_SIZE": Config.BATCH_SIZE,
                "HIDDEN_SIZE": Config.HIDDEN_SIZE,
                "NUM_LAYERS": Config.NUM_LAYERS,
                "NUM_HEADS": Config.NUM_HEADS,
                "DROPOUT": Config.DROPOUT,
                "LR": Config.LR,
                "EPOCHS": Config.EPOCHS,
                "MODEL_TYPE": Config.MODEL_TYPE,
                "FLARE_THRESHOLD": Config.FLARE_THRESHOLD,
                "TASK_WEIGHTS": Config.TASK_WEIGHTS
            }
            
            torch.save({
                "model": model.state_dict(),
                "scaler_center": scaler.center_,
                "scaler_scale": scaler.scale_,
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "config": config_dict
            }, model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch} | Train Loss: {train_total_loss:.4f} | "
                  f"Val Loss: {val_total_loss:.4f} | Val Acc: {val_cls_acc:.3f}")

    print(f"✅ Saved multi-task model to {model_path}")
    return True


# ================================
# SHAP EXPLAINABILITY
# ================================
class SHAPExplainer:
    def __init__(self, model, scaler, feature_cols):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        
    def explain_prediction(self, df, target_col="symptoms.gi_flare"):
        """Generate SHAP explanations for the last prediction"""
        try:
            # Prepare data
            df_scaled = df.copy()
            df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
            
            # Get last sequence
            last_seq = df_scaled.tail(Config.SEQ_LEN)[self.feature_cols].values.astype(np.float32)
            if len(last_seq) < Config.SEQ_LEN:
                return None
            
            # Create SHAP explainer
            def model_predict(X):
                X_tensor = torch.tensor(X, dtype=torch.float32)
                with torch.no_grad():
                    reg_pred, cls_pred = self.model(X_tensor)
                return reg_pred.numpy()
            
            explainer = shap.Explainer(model_predict, last_seq)
            shap_values = explainer(last_seq)
            
            # Get feature importance
            feature_importance = np.abs(shap_values.values).mean(axis=0)
            feature_names = [f"{col}_t{i}" for col in self.feature_cols for i in range(Config.SEQ_LEN)]
            
            # Get top contributing features
            top_features = sorted(zip(feature_names, feature_importance), 
                                key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "shap_values": shap_values.values,
                "top_features": top_features,
                "prediction": model_predict(last_seq)[0]
            }
        except Exception as e:
            print(f"SHAP explanation error: {e}")
            return None


# ================================
# ENHANCED INFERENCE & ALERTS
# ================================
def predict_next_multitask(df, model_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(model_path):
        return None, None

    try:
        ckpt = torch.load(model_path, map_location=DEVICE)
        feature_cols = ckpt["feature_cols"]
        target_cols = ckpt["target_cols"]
        
        scaler = RobustScaler()
        scaler.center_ = ckpt["scaler_center"]
        scaler.scale_ = ckpt["scaler_scale"]

        df_ = df.copy()
        df_[feature_cols] = scaler.transform(df_[feature_cols])
        seq = df_.tail(Config.SEQ_LEN)[feature_cols].values.astype(np.float32)
        
        if len(seq) < Config.SEQ_LEN:
            return None, None
            
        x = torch.tensor(seq).unsqueeze(0).to(DEVICE)

        model = TransformerRegressor(len(feature_cols), Config.HIDDEN_SIZE,
                                   Config.NUM_HEADS, Config.NUM_LAYERS, 
                                   Config.DROPOUT, Config.SEQ_LEN).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        
        with torch.no_grad():
            reg_pred, cls_pred = model(x)
            reg_value = reg_pred.item()
            cls_prob = torch.sigmoid(cls_pred).item()
            
        return reg_value, cls_prob
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

def generate_enhanced_alerts(df, preds, explainer=None):
    alerts = []
    if df.empty: 
        return alerts
    
    last = df.tail(1)
    sleep = float(last["sleep.duration_hours"].iloc[0])
    stress = float(last["mood.stress_score"].iloc[0])
    
    # Rule-based alerts
    if sleep < Config.LOW_SLEEP_HOURS:
        alerts.append(f"⚠️ Low sleep ({sleep:.1f}h) - aim for 7-9 hours")
    if stress >= Config.HIGH_STRESS:
        alerts.append(f"⚠️ High stress ({stress:.1f}/10) - consider relaxation techniques")
    
    # Model-based alerts with explanations
    if preds.get("gi_flare_reg") is not None and preds.get("gi_flare_cls") is not None:
        reg_value = preds["gi_flare_reg"]
        cls_prob = preds["gi_flare_cls"]
        
        if cls_prob > 0.7:  # High probability of flare
            alerts.append(f"🔮 High GI flare risk tomorrow ({cls_prob:.1%} probability, ~{reg_value:.1f}/10)")
            
            # Add SHAP explanation if available
            if explainer:
                explanation = explainer.explain_prediction(df)
                if explanation and explanation["top_features"]:
                    top_feature = explanation["top_features"][0]
                    alerts.append(f"   📊 Key factor: {top_feature[0]} (impact: {top_feature[1]:.3f})")
    
    if preds.get("mood_reg") is not None:
        mood_pred = preds["mood_reg"]
        if mood_pred <= 4.0:
            alerts.append(f"🔮 Predicted low mood tomorrow (~{mood_pred:.1f}/10)")

    return alerts


# ================================
# PERSONALIZED INSIGHTS SERVICE
# ================================
class InsightsService:
    def __init__(self):
        self.insights_cache = {}
    
    def generate_insights(self, health_data):
        """Generate personalized insights from health data"""
        insights = []
        
        if len(health_data) < 7:
            return insights
        
        # Sort data by date (handle both indexed and non-indexed DataFrames)
        if 'date' in health_data.columns:
            sorted_data = health_data.sort_values('date')
        else:
            # DataFrame is already indexed by date, just sort by index
            sorted_data = health_data.sort_index()
        
        # Generate different types of insights
        insights.extend(self._analyze_sleep_patterns(sorted_data))
        insights.extend(self._analyze_mood_trends(sorted_data))
        insights.extend(self._analyze_stress_patterns(sorted_data))
        insights.extend(self._analyze_symptom_correlations(sorted_data))
        insights.extend(self._analyze_lifestyle_factors(sorted_data))
        
        # Sort by priority and confidence
        return sorted(insights, key=lambda x: (x['priority'], x['confidence']), reverse=True)
    
    def _analyze_sleep_patterns(self, data):
        insights = []
        sleep_data = data[data['sleep.duration_hours'].notna() & data['sleep.quality_score'].notna()]
        
        if len(sleep_data) < 7:
            return insights
        
        # Analyze sleep duration trends
        durations = sleep_data['sleep.duration_hours'].values
        avg_duration = np.mean(durations)
        recent_avg = np.mean(durations[-7:])
        
        if avg_duration < 6.5:
            insights.append({
                'id': 'sleep_duration_low',
                'type': 'sleep',
                'title': 'Sleep Duration Alert',
                'description': f'Your average sleep duration is {avg_duration:.1f} hours, which is below the recommended 7-9 hours.',
                'actionable': True,
                'priority': 'high',
                'confidence': 0.9,
                'recommendations': [
                    'Try going to bed 30 minutes earlier',
                    'Create a consistent bedtime routine',
                    'Avoid screens 1 hour before bed',
                    'Keep your bedroom cool and dark'
                ],
                'category': 'warning'
            })
        
        # Analyze sleep quality trends
        qualities = sleep_data['sleep.quality_score'].values
        avg_quality = np.mean(qualities)
        recent_quality = np.mean(qualities[-7:])
        
        if recent_quality > avg_quality + 1:
            insights.append({
                'id': 'sleep_quality_improving',
                'type': 'sleep',
                'title': 'Sleep Quality Improving!',
                'description': f'Your sleep quality has improved from {avg_quality:.1f} to {recent_quality:.1f} over the past week.',
                'actionable': False,
                'priority': 'low',
                'confidence': 0.8,
                'category': 'achievement'
            })
        
        return insights
    
    def _analyze_mood_trends(self, data):
        insights = []
        mood_data = data[data['mood.mood_score'].notna()]
        
        if len(mood_data) < 7:
            return insights
        
        moods = mood_data['mood.mood_score'].values
        avg_mood = np.mean(moods)
        recent_mood = np.mean(moods[-7:])
        
        if recent_mood < avg_mood - 1.5:
            insights.append({
                'id': 'mood_declining',
                'type': 'mood',
                'title': 'Mood Trend Alert',
                'description': f'Your mood has declined from {avg_mood:.1f} to {recent_mood:.1f} over the past week.',
                'actionable': True,
                'priority': 'high',
                'confidence': 0.8,
                'recommendations': [
                    'Consider talking to a mental health professional',
                    'Try daily gratitude journaling',
                    'Increase physical activity',
                    'Connect with friends and family'
                ],
                'category': 'warning'
            })
        
        return insights
    
    def _analyze_stress_patterns(self, data):
        insights = []
        stress_data = data[data['mood.stress_score'].notna()]
        
        if len(stress_data) < 7:
            return insights
        
        stress_levels = stress_data['mood.stress_score'].values
        avg_stress = np.mean(stress_levels)
        
        if avg_stress > 7:
            insights.append({
                'id': 'high_stress',
                'type': 'stress',
                'title': 'High Stress Levels',
                'description': f'Your average stress level is {avg_stress:.1f}/10, which is quite high.',
                'actionable': True,
                'priority': 'high',
                'confidence': 0.9,
                'recommendations': [
                    'Practice deep breathing exercises',
                    'Try progressive muscle relaxation',
                    'Consider stress management therapy',
                    'Identify and address stress sources'
                ],
                'category': 'warning'
            })
        
        return insights
    
    def _analyze_symptom_correlations(self, data):
        insights = []
        
        # Analyze sleep-symptom correlations
        sleep_qualities = data[data['sleep.quality_score'].notna()]['sleep.quality_score'].values
        gi_flares = data[data['symptoms.gi_flare'].notna()]['symptoms.gi_flare'].values
        
        if len(sleep_qualities) > 0 and len(gi_flares) > 0:
            correlation = np.corrcoef(sleep_qualities, gi_flares)[0, 1]
            
            if correlation < -0.4:
                insights.append({
                    'id': 'sleep_gi_correlation',
                    'type': 'correlation',
                    'title': 'Sleep Quality & GI Symptoms',
                    'description': 'Poor sleep quality is correlated with increased GI flare-ups in your data.',
                    'actionable': True,
                    'priority': 'medium',
                    'confidence': 0.7,
                    'recommendations': [
                        'Prioritize sleep quality to reduce GI symptoms',
                        'Try sleep hygiene improvements',
                        'Consider sleep environment optimization'
                    ],
                    'category': 'correlation'
                })
        
        return insights
    
    def _analyze_lifestyle_factors(self, data):
        insights = []
        
        # Analyze weekend vs weekday patterns
        # Handle both indexed and non-indexed DataFrames
        if 'date' in data.columns:
            dates = pd.to_datetime(data['date'])
        else:
            # DataFrame is indexed by date
            dates = data.index
        
        data_copy = data.copy()
        data_copy['day_of_week'] = dates.dayofweek
        
        weekend_data = data_copy[data_copy['day_of_week'].isin([5, 6])]  # Saturday, Sunday
        weekday_data = data_copy[data_copy['day_of_week'].isin([0, 1, 2, 3, 4])]  # Monday-Friday
        
        if len(weekend_data) > 0 and len(weekday_data) > 0:
            weekend_sleep = weekend_data[weekend_data['sleep.duration_hours'].notna()]['sleep.duration_hours'].values
            weekday_sleep = weekday_data[weekday_data['sleep.duration_hours'].notna()]['sleep.duration_hours'].values
            
            if len(weekend_sleep) > 0 and len(weekday_sleep) > 0:
                weekend_avg = np.mean(weekend_sleep)
                weekday_avg = np.mean(weekday_sleep)
                
                if weekend_avg > weekday_avg + 1:
                    insights.append({
                        'id': 'weekend_sleep_catchup',
                        'type': 'sleep',
                        'title': 'Weekend Sleep Catch-up',
                        'description': f'You sleep {weekend_avg - weekday_avg:.1f} hours more on weekends, suggesting weekday sleep debt.',
                        'actionable': True,
                        'priority': 'medium',
                        'confidence': 0.8,
                        'recommendations': [
                            'Try to get more consistent sleep during weekdays',
                            'Consider adjusting weekday bedtime',
                            'Avoid oversleeping on weekends'
                        ],
                        'category': 'tip'
                    })
        
        return insights


# ================================
# COPING STRATEGIES SERVICE
# ================================
class CopingStrategiesService:
    def __init__(self):
        self.strategies = self._initialize_strategies()
        self.mood_boosters = self._initialize_mood_boosters()
    
    def _initialize_strategies(self):
        return [
            {
                'id': 'box_breathing',
                'name': 'Box Breathing',
                'description': 'A simple breathing technique to quickly reduce stress and anxiety',
                'category': 'breathing',
                'duration': 5,
                'difficulty': 'beginner',
                'effectiveness': 4,
                'instructions': [
                    'Sit comfortably with your back straight',
                    'Inhale slowly for 4 counts',
                    'Hold your breath for 4 counts',
                    'Exhale slowly for 4 counts',
                    'Hold empty for 4 counts',
                    'Repeat for 5-10 cycles'
                ],
                'benefits': [
                    'Reduces stress and anxiety',
                    'Improves focus and concentration',
                    'Activates the parasympathetic nervous system',
                    'Can be done anywhere, anytime'
                ],
                'when_to_use': [
                    'Before important meetings or presentations',
                    'When feeling overwhelmed',
                    'During panic attacks or anxiety episodes',
                    'Before bedtime to promote relaxation'
                ]
            },
            {
                'id': 'progressive_muscle_relaxation',
                'name': 'Progressive Muscle Relaxation',
                'description': 'Systematically tense and relax muscle groups to release physical tension',
                'category': 'physical',
                'duration': 15,
                'difficulty': 'beginner',
                'effectiveness': 4,
                'instructions': [
                    'Find a quiet, comfortable place to lie down',
                    'Start with your toes - tense for 5 seconds, then relax',
                    'Move up to your calves, thighs, and glutes',
                    'Continue with your abdomen, chest, and back',
                    'Tense and relax your arms, hands, and fingers',
                    'Finish with your neck, face, and jaw',
                    'Take deep breaths and notice the relaxation'
                ],
                'benefits': [
                    'Reduces muscle tension and pain',
                    'Improves sleep quality',
                    'Decreases anxiety and stress',
                    'Increases body awareness'
                ],
                'when_to_use': [
                    'When experiencing muscle tension or pain',
                    'Before bedtime for better sleep',
                    'After a stressful day',
                    'When feeling physically tense'
                ]
            },
            {
                'id': 'mindful_breathing',
                'name': 'Mindful Breathing',
                'description': 'Focus on your breath to anchor yourself in the present moment',
                'category': 'mindfulness',
                'duration': 10,
                'difficulty': 'beginner',
                'effectiveness': 3,
                'instructions': [
                    'Sit or lie in a comfortable position',
                    'Close your eyes or soften your gaze',
                    'Focus on your natural breathing rhythm',
                    'Notice the sensation of air entering and leaving your nostrils',
                    'When your mind wanders, gently return to your breath',
                    'Continue for 10 minutes or as long as comfortable'
                ],
                'benefits': [
                    'Reduces stress and anxiety',
                    'Improves focus and attention',
                    'Increases emotional regulation',
                    'Promotes relaxation and calm'
                ],
                'when_to_use': [
                    'When feeling scattered or unfocused',
                    'During moments of high stress',
                    'As a daily mindfulness practice',
                    'When you need to center yourself'
                ]
            }
        ]
    
    def _initialize_mood_boosters(self):
        return [
            {
                'id': 'nature_walk',
                'name': 'Nature Walk',
                'description': 'Take a walk in nature to boost your mood and energy',
                'category': 'nature',
                'duration': 20,
                'mood_impact': 4,
                'energy_level': 'medium',
                'instructions': [
                    'Find a nearby park, trail, or green space',
                    'Leave your phone behind or put it on silent',
                    'Walk at a comfortable pace',
                    'Notice the sights, sounds, and smells around you',
                    'Take deep breaths of fresh air',
                    'Spend at least 20 minutes outdoors'
                ],
                'tips': [
                    'Even a 10-minute walk can help',
                    'Try to go during daylight hours',
                    'Bring water and wear comfortable shoes',
                    'Focus on being present rather than exercising'
                ]
            },
            {
                'id': 'music_therapy',
                'name': 'Music Therapy',
                'description': 'Listen to uplifting music to improve your mood',
                'category': 'activity',
                'duration': 15,
                'mood_impact': 4,
                'energy_level': 'low',
                'instructions': [
                    'Choose music that makes you feel good',
                    'Create a playlist of your favorite uplifting songs',
                    'Find a comfortable place to listen',
                    'Close your eyes and focus on the music',
                    'Allow yourself to feel the emotions the music evokes',
                    'Consider singing along or moving to the beat'
                ],
                'tips': [
                    'Upbeat, major-key songs tend to be most uplifting',
                    'Avoid sad or angry music when feeling down',
                    'Try different genres to find what works for you',
                    'Consider creating different playlists for different moods'
                ]
            },
            {
                'id': 'gratitude_journal',
                'name': 'Gratitude Journaling',
                'description': 'Write down things you\'re grateful for to shift your perspective',
                'category': 'mindfulness',
                'duration': 10,
                'mood_impact': 3,
                'energy_level': 'low',
                'instructions': [
                    'Get a notebook or use a journaling app',
                    'Write down 3-5 things you\'re grateful for today',
                    'Be specific and detailed in your descriptions',
                    'Include both big and small things',
                    'Reflect on why you\'re grateful for each item',
                    'Read back over previous entries when feeling down'
                ],
                'tips': [
                    'Try to write at the same time each day',
                    'Don\'t worry about being profound - simple things count',
                    'Include people, experiences, and even challenges',
                    'Consider sharing your gratitude with others'
                ]
            }
        ]
    
    def get_recommended_strategies(self, current_mood, stress_level, energy_level):
        """Get recommended coping strategies based on current state"""
        recommended = []
        
        for strategy in self.strategies:
            # Filter strategies based on current state
            if current_mood <= 3 and strategy['effectiveness'] >= 4:
                recommended.append(strategy)
            elif stress_level >= 8 and strategy['duration'] <= 10:
                recommended.append(strategy)
            elif energy_level <= 3 and strategy['category'] != 'physical':
                recommended.append(strategy)
            else:
                recommended.append(strategy)
        
        # Sort by effectiveness and duration
        return sorted(recommended, key=lambda x: (x['effectiveness'], -x['duration']), reverse=True)
    
    def get_recommended_mood_boosters(self, current_mood, energy_level):
        """Get recommended mood boosters based on current state"""
        recommended = []
        
        for booster in self.mood_boosters:
            # Filter based on current state
            if current_mood <= 3 and booster['mood_impact'] >= 4:
                recommended.append(booster)
            elif energy_level <= 3 and booster['energy_level'] != 'high':
                recommended.append(booster)
            else:
                recommended.append(booster)
        
        # Sort by mood impact
        return sorted(recommended, key=lambda x: x['mood_impact'], reverse=True)


# ================================
# GAMIFICATION SERVICE
# ================================
class GamificationService:
    def __init__(self):
        self.badges = self._initialize_badges()
        self.achievements = self._initialize_achievements()
    
    def _initialize_badges(self):
        return [
            {
                'id': 'streak_3',
                'name': 'Getting Started',
                'description': 'Log 3 days in a row',
                'icon': 'flame',
                'color': '#FF6B6B',
                'category': 'streak',
                'progress': 0,
                'max_progress': 3,
                'is_unlocked': False
            },
            {
                'id': 'streak_7',
                'name': 'Week Warrior',
                'description': 'Log 7 days in a row',
                'icon': 'trophy',
                'color': '#4ECDC4',
                'category': 'streak',
                'progress': 0,
                'max_progress': 7,
                'is_unlocked': False
            },
            {
                'id': 'streak_30',
                'name': 'Monthly Master',
                'description': 'Log 30 days in a row',
                'icon': 'medal',
                'color': '#45B7D1',
                'category': 'streak',
                'progress': 0,
                'max_progress': 30,
                'is_unlocked': False
            },
            {
                'id': 'milestone_50',
                'name': 'Half Century',
                'description': 'Complete 50 health logs',
                'icon': 'checkmark-circle',
                'color': '#FF6348',
                'category': 'milestone',
                'progress': 0,
                'max_progress': 50,
                'is_unlocked': False
            },
            {
                'id': 'mood_improvement',
                'name': 'Mood Booster',
                'description': 'Improve mood score by 2+ points over 7 days',
                'icon': 'happy',
                'color': '#54A0FF',
                'category': 'improvement',
                'progress': 0,
                'max_progress': 1,
                'is_unlocked': False
            }
        ]
    
    def _initialize_achievements(self):
        return [
            {
                'id': 'first_log',
                'title': 'First Steps',
                'description': 'Complete your first health log',
                'points': 10,
                'is_unlocked': False
            },
            {
                'id': 'week_complete',
                'title': 'Week Complete',
                'description': 'Log every day for a full week',
                'points': 50,
                'is_unlocked': False
            },
            {
                'id': 'mood_master',
                'title': 'Mood Master',
                'description': 'Maintain average mood above 8 for a week',
                'points': 100,
                'is_unlocked': False
            }
        ]
    
    def update_stats(self, health_data):
        """Update user stats and check for badge/achievement unlocks"""
        stats = self._calculate_stats(health_data)
        self._check_badges(health_data, stats)
        self._check_achievements(health_data, stats)
        return stats
    
    def _calculate_stats(self, health_data):
        """Calculate user statistics"""
        total_logs = len(health_data)
        # Handle both indexed and non-indexed DataFrames
        if 'date' in health_data.columns:
            total_days = len(health_data['date'].unique())
        else:
            total_days = len(health_data.index.unique())
        
        # Calculate averages
        mood_scores = health_data[health_data['mood.mood_score'].notna()]['mood.mood_score'].values
        sleep_scores = health_data[health_data['sleep.quality_score'].notna()]['sleep.quality_score'].values
        stress_scores = health_data[health_data['mood.stress_score'].notna()]['mood.stress_score'].values
        
        average_mood = np.mean(mood_scores) if len(mood_scores) > 0 else 0
        average_sleep = np.mean(sleep_scores) if len(sleep_scores) > 0 else 0
        average_stress = np.mean(stress_scores) if len(stress_scores) > 0 else 0
        
        # Calculate streaks
        streaks = self._calculate_streaks(health_data)
        
        # Calculate level and experience
        experience = total_logs * 10  # 10 XP per log
        level = experience // 100 + 1  # 100 XP per level
        next_level_exp = level * 100
        
        return {
            'total_logs': total_logs,
            'total_days': total_days,
            'average_mood': average_mood,
            'average_sleep': average_sleep,
            'average_stress': average_stress,
            'badges': self.badges,
            'achievements': self.achievements,
            'streaks': streaks,
            'level': level,
            'experience': experience,
            'next_level_exp': next_level_exp
        }
    
    def _calculate_streaks(self, health_data):
        """Calculate current and longest streaks"""
        if len(health_data) == 0:
            return {'current_streak': 0, 'longest_streak': 0, 'last_log_date': ''}
        
        # Handle both indexed and non-indexed DataFrames
        if 'date' in health_data.columns:
            sorted_data = health_data.sort_values('date')
            dates = sorted_data['date'].unique()
        else:
            # DataFrame is indexed by date
            sorted_data = health_data.sort_index()
            dates = sorted_data.index.unique()
        
        # Calculate current streak
        current_streak = 0
        today = pd.Timestamp.now().date()
        
        for i in range(len(dates)):
            date_diff = (today - pd.Timestamp(dates[-(i+1)]).date()).days
            if date_diff == i:
                current_streak += 1
            else:
                break
        
        # Calculate longest streak
        longest_streak = 0
        temp_streak = 1
        
        for i in range(1, len(dates)):
            date_diff = (pd.Timestamp(dates[i]).date() - pd.Timestamp(dates[i-1]).date()).days
            if date_diff == 1:
                temp_streak += 1
            else:
                longest_streak = max(longest_streak, temp_streak)
                temp_streak = 1
        
        longest_streak = max(longest_streak, temp_streak)
        
        return {
            'current_streak': current_streak,
            'longest_streak': longest_streak,
            'last_log_date': str(dates[-1]) if len(dates) > 0 else ''
        }
    
    def _check_badges(self, health_data, stats):
        """Check for badge unlocks"""
        for badge in self.badges:
            if badge['is_unlocked']:
                continue
            
            should_unlock = False
            
            if badge['id'].startswith('streak_'):
                required_streak = int(badge['id'].split('_')[1])
                badge['progress'] = min(stats['streaks']['current_streak'], badge['max_progress'])
                should_unlock = stats['streaks']['current_streak'] >= required_streak
            elif badge['id'].startswith('milestone_'):
                required_logs = int(badge['id'].split('_')[1])
                badge['progress'] = min(stats['total_logs'], badge['max_progress'])
                should_unlock = stats['total_logs'] >= required_logs
            
            if should_unlock:
                badge['is_unlocked'] = True
                badge['unlocked_at'] = datetime.now().isoformat()
                print(f"🎉 Badge unlocked: {badge['name']} - {badge['description']}")
    
    def _check_achievements(self, health_data, stats):
        """Check for achievement unlocks"""
        for achievement in self.achievements:
            if achievement['is_unlocked']:
                continue
            
            should_unlock = False
            
            if achievement['id'] == 'first_log':
                should_unlock = stats['total_logs'] >= 1
            elif achievement['id'] == 'week_complete':
                should_unlock = stats['streaks']['current_streak'] >= 7
            elif achievement['id'] == 'mood_master':
                should_unlock = stats['average_mood'] >= 8
            
            if should_unlock:
                achievement['is_unlocked'] = True
                achievement['unlocked_at'] = datetime.now().isoformat()
                print(f"🏆 Achievement unlocked: {achievement['title']} - {achievement['description']}")


# ================================
# TREND ANALYSIS SERVICE
# ================================
class TrendAnalysisService:
    def __init__(self):
        pass
    
    def generate_trend_charts(self, health_data, time_range='week'):
        """Generate trend analysis charts"""
        if len(health_data) < 7:
            return None
        
        # Filter data based on time range
        filtered_data = self._filter_data_by_time_range(health_data, time_range)
        
        if len(filtered_data) == 0:
            return None
        
        # Sort by date
        sorted_data = filtered_data.sort_values('date')
        
        # Generate chart data
        chart_data = {
            'labels': self._generate_labels(sorted_data, time_range),
            'sleep_duration': sorted_data['sleep.duration_hours'].fillna(0).values,
            'sleep_quality': sorted_data['sleep.quality_score'].fillna(0).values,
            'mood_score': sorted_data['mood.mood_score'].fillna(0).values,
            'stress_score': sorted_data['mood.stress_score'].fillna(0).values,
            'gi_flare': sorted_data['symptoms.gi_flare'].fillna(0).values
        }
        
        return chart_data
    
    def _filter_data_by_time_range(self, data, time_range):
        """Filter data based on time range"""
        now = pd.Timestamp.now()
        
        if time_range == 'week':
            cutoff_date = now - pd.Timedelta(days=7)
        elif time_range == 'month':
            cutoff_date = now - pd.Timedelta(days=30)
        elif time_range == 'quarter':
            cutoff_date = now - pd.Timedelta(days=90)
        else:
            return data
        
        # Handle both indexed and non-indexed DataFrames
        if 'date' in data.columns:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])
            return data_copy[data_copy['date'] >= cutoff_date]
        else:
            # DataFrame is indexed by date
            return data[data.index >= cutoff_date]
    
    def _generate_labels(self, data, time_range):
        """Generate labels for charts"""
        if len(data) == 0:
            return []
        
        # Handle both indexed and non-indexed DataFrames
        if 'date' in data.columns:
            dates = data['date']
        else:
            dates = data.index
        
        if time_range == 'week':
            return [pd.Timestamp(date).strftime('%a') for date in dates]
        elif time_range == 'month':
            return [pd.Timestamp(date).strftime('%m/%d') for date in dates]
        else:
            return [pd.Timestamp(date).strftime('%m/%d') for date in dates]
    
    def generate_correlation_heatmap(self, health_data, metric='gi_flare'):
        """Generate correlation heatmap data"""
        if len(health_data) < 14:
            return None
        
        # Prepare data for correlation analysis
        correlation_data = health_data[['sleep.duration_hours', 'sleep.quality_score', 
                                      'mood.mood_score', 'mood.stress_score',
                                      'symptoms.gi_flare', 'symptoms.skin_flare', 
                                      'symptoms.migraine']].fillna(0)
        
        # Calculate correlation matrix
        correlation_matrix = correlation_data.corr()
        
        # Generate heatmap data
        heatmap_data = {
            'correlations': correlation_matrix.values.tolist(),
            'labels': list(correlation_matrix.columns),
            'metric': metric
        }
        
        return heatmap_data


# ================================
# VOICE TRANSCRIPTION SIMULATION
# ================================
class VoiceTranscriptionService:
    def __init__(self):
        self.transcription_templates = [
            "I had a really tough day today...",
            "My sleep was terrible last night...",
            "I feel really stressed about work...",
            "My mood has been up and down...",
            "I think I need to take better care of myself...",
            "Feeling grateful for the small things today...",
            "Had a great conversation with a friend...",
            "Work is getting overwhelming lately...",
            "My symptoms seem to be flaring up...",
            "Feeling more positive about the future..."
        ]
    
    def simulate_transcription(self, duration_seconds=30):
        """Simulate voice transcription based on duration"""
        # Simulate transcription based on duration
        if duration_seconds < 10:
            return random.choice(self.transcription_templates[:3])
        elif duration_seconds < 20:
            return random.choice(self.transcription_templates[3:7])
        else:
            return random.choice(self.transcription_templates[7:])
    
    def analyze_sentiment(self, text):
        """Simple sentiment analysis of transcribed text"""
        positive_words = ['good', 'great', 'happy', 'grateful', 'positive', 'better', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'stressed', 'overwhelming', 'difficult', 'tough', 'worried']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'


# ================================
# SOCIAL & SHARING SERVICE
# ================================
class SocialSharingService:
    def __init__(self):
        self.community_benchmarks = self._initialize_benchmarks()
        self.sharing_templates = self._initialize_sharing_templates()
    
    def _initialize_benchmarks(self):
        """Initialize community benchmark data"""
        return {
            'sleep_quality': {
                'mean': 6.8,
                'std': 1.2,
                'sample_size': 5000,
                'percentiles': {
                    25: 6.0,
                    50: 6.8,
                    75: 7.5,
                    90: 8.2
                }
            },
            'sleep_duration': {
                'mean': 7.2,
                'std': 1.1,
                'sample_size': 5000,
                'percentiles': {
                    25: 6.5,
                    50: 7.2,
                    75: 7.8,
                    90: 8.5
                }
            },
            'stress_levels': {
                'mean': 5.4,
                'std': 2.1,
                'sample_size': 5000,
                'percentiles': {
                    25: 3.5,
                    50: 5.4,
                    75: 7.2,
                    90: 8.5
                }
            },
            'mood_scores': {
                'mean': 6.5,
                'std': 1.8,
                'sample_size': 5000,
                'percentiles': {
                    25: 5.2,
                    50: 6.5,
                    75: 7.8,
                    90: 8.7
                }
            }
        }
    
    def _initialize_sharing_templates(self):
        """Initialize sharing report templates"""
        return {
            'weekly_summary': {
                'title': 'Weekly Health Summary',
                'sections': ['sleep', 'mood', 'stress', 'symptoms', 'insights'],
                'format': 'pdf'
            },
            'medical_report': {
                'title': 'Medical Health Report',
                'sections': ['sleep', 'symptoms', 'trends', 'correlations'],
                'format': 'pdf'
            },
            'therapist_summary': {
                'title': 'Mental Health Summary',
                'sections': ['mood', 'stress', 'sleep', 'journal_highlights'],
                'format': 'pdf'
            }
        }
    
    def generate_community_benchmark(self, user_data, metric):
        """Generate community benchmark comparison"""
        if metric not in self.community_benchmarks:
            return None
        
        benchmark = self.community_benchmarks[metric]
        
        # Calculate user's average
        if metric == 'sleep_quality':
            user_avg = user_data['sleep.quality_score'].mean() if 'sleep.quality_score' in user_data.columns else 0
        elif metric == 'sleep_duration':
            user_avg = user_data['sleep.duration_hours'].mean() if 'sleep.duration_hours' in user_data.columns else 0
        elif metric == 'stress_levels':
            user_avg = user_data['mood.stress_score'].mean() if 'mood.stress_score' in user_data.columns else 0
        elif metric == 'mood_scores':
            user_avg = user_data['mood.mood_score'].mean() if 'mood.mood_score' in user_data.columns else 0
        else:
            return None
        
        if user_avg == 0:
            return None
        
        # Calculate percentile
        community_mean = benchmark['mean']
        community_std = benchmark['std']
        
        # Calculate z-score
        z_score = (user_avg - community_mean) / community_std
        
        # Convert to percentile
        percentile = 50 + (z_score * 20)  # Rough approximation
        percentile = max(1, min(99, percentile))
        
        # Calculate percentage difference
        percentage_diff = ((user_avg - community_mean) / community_mean) * 100
        
        return {
            'metric': metric,
            'user_average': user_avg,
            'community_average': community_mean,
            'percentage_difference': percentage_diff,
            'percentile': percentile,
            'sample_size': benchmark['sample_size'],
            'message': self._generate_benchmark_message(metric, percentage_diff, percentile)
        }
    
    def _generate_benchmark_message(self, metric, percentage_diff, percentile):
        """Generate personalized benchmark message"""
        if abs(percentage_diff) < 5:
            return f"Your {metric.replace('_', ' ')} is very close to the community average!"
        elif percentage_diff > 0:
            return f"Compared to {self.community_benchmarks[metric]['sample_size']:,} other users, your {metric.replace('_', ' ')} is {abs(percentage_diff):.0f}% higher (top {100-percentile:.0f}%)!"
        else:
            return f"Compared to {self.community_benchmarks[metric]['sample_size']:,} other users, your {metric.replace('_', ' ')} is {abs(percentage_diff):.0f}% lower (bottom {percentile:.0f}%)."
    
    def generate_export_report(self, user_data, report_type, time_range='3_months'):
        """Generate exportable report"""
        if report_type not in self.sharing_templates:
            return None
        
        template = self.sharing_templates[report_type]
        
        # Filter data by time range
        filtered_data = self._filter_data_by_time_range(user_data, time_range)
        
        if len(filtered_data) == 0:
            return None
        
        # Generate report sections
        report = {
            'title': template['title'],
            'generated_at': datetime.now().isoformat(),
            'time_range': time_range,
            'sections': {}
        }
        
        # Generate each section
        for section in template['sections']:
            report['sections'][section] = self._generate_section_data(filtered_data, section)
        
        return report
    
    def _filter_data_by_time_range(self, data, time_range):
        """Filter data by time range"""
        now = pd.Timestamp.now()
        
        if time_range == '1_week':
            cutoff_date = now - pd.Timedelta(days=7)
        elif time_range == '1_month':
            cutoff_date = now - pd.Timedelta(days=30)
        elif time_range == '3_months':
            cutoff_date = now - pd.Timedelta(days=90)
        elif time_range == '6_months':
            cutoff_date = now - pd.Timedelta(days=180)
        else:
            return data
        
        # Handle both indexed and non-indexed DataFrames
        if 'date' in data.columns:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])
            return data_copy[data_copy['date'] >= cutoff_date]
        else:
            return data[data.index >= cutoff_date]
    
    def _generate_section_data(self, data, section):
        """Generate data for specific report section"""
        if section == 'sleep':
            return {
                'average_duration': data['sleep.duration_hours'].mean() if 'sleep.duration_hours' in data.columns else 0,
                'average_quality': data['sleep.quality_score'].mean() if 'sleep.quality_score' in data.columns else 0,
                'consistency_score': self._calculate_consistency(data, 'sleep.duration_hours'),
                'trend': self._calculate_trend(data, 'sleep.quality_score')
            }
        elif section == 'mood':
            return {
                'average_mood': data['mood.mood_score'].mean() if 'mood.mood_score' in data.columns else 0,
                'average_stress': data['mood.stress_score'].mean() if 'mood.stress_score' in data.columns else 0,
                'mood_stability': self._calculate_consistency(data, 'mood.mood_score'),
                'trend': self._calculate_trend(data, 'mood.mood_score')
            }
        elif section == 'symptoms':
            return {
                'gi_flare_avg': data['symptoms.gi_flare'].mean() if 'symptoms.gi_flare' in data.columns else 0,
                'skin_flare_avg': data['symptoms.skin_flare'].mean() if 'symptoms.skin_flare' in data.columns else 0,
                'migraine_avg': data['symptoms.migraine'].mean() if 'symptoms.migraine' in data.columns else 0,
                'flare_frequency': self._calculate_flare_frequency(data)
            }
        elif section == 'insights':
            return {
                'key_correlations': self._find_key_correlations(data),
                'improvement_areas': self._identify_improvement_areas(data),
                'strengths': self._identify_strengths(data)
            }
        else:
            return {}
    
    def _calculate_consistency(self, data, column):
        """Calculate consistency score (lower std = more consistent)"""
        if column not in data.columns:
            return 0
        values = data[column].dropna()
        if len(values) < 2:
            return 0
        return 1 / (1 + values.std())  # Convert std to consistency score
    
    def _calculate_trend(self, data, column):
        """Calculate trend direction"""
        if column not in data.columns:
            return 'stable'
        values = data[column].dropna()
        if len(values) < 3:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'improving'
        elif slope < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_flare_frequency(self, data):
        """Calculate flare frequency"""
        flare_columns = ['symptoms.gi_flare', 'symptoms.skin_flare', 'symptoms.migraine']
        total_flares = 0
        total_days = len(data)
        
        for col in flare_columns:
            if col in data.columns:
                total_flares += (data[col] > 5).sum()  # Flare threshold
        
        return total_flares / (total_days * len(flare_columns)) if total_days > 0 else 0
    
    def _find_key_correlations(self, data):
        """Find key correlations in the data"""
        correlations = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr = data[col1].corr(data[col2])
                if abs(corr) > 0.5:  # Strong correlation
                    correlations.append({
                        'metric1': col1,
                        'metric2': col2,
                        'correlation': corr,
                        'strength': 'strong' if abs(corr) > 0.7 else 'moderate'
                    })
        
        return correlations[:5]  # Top 5 correlations
    
    def _identify_improvement_areas(self, data):
        """Identify areas for improvement"""
        areas = []
        
        if 'sleep.duration_hours' in data.columns:
            avg_sleep = data['sleep.duration_hours'].mean()
            if avg_sleep < 7:
                areas.append(f"Sleep duration (currently {avg_sleep:.1f}h, aim for 7-9h)")
        
        if 'mood.stress_score' in data.columns:
            avg_stress = data['mood.stress_score'].mean()
            if avg_stress > 7:
                areas.append(f"Stress management (currently {avg_stress:.1f}/10, aim for <7)")
        
        if 'sleep.quality_score' in data.columns:
            avg_quality = data['sleep.quality_score'].mean()
            if avg_quality < 6:
                areas.append(f"Sleep quality (currently {avg_quality:.1f}/10, aim for >6)")
        
        return areas
    
    def _identify_strengths(self, data):
        """Identify user strengths"""
        strengths = []
        
        if 'sleep.duration_hours' in data.columns:
            avg_sleep = data['sleep.duration_hours'].mean()
            if avg_sleep >= 7.5:
                strengths.append(f"Excellent sleep duration ({avg_sleep:.1f}h)")
        
        if 'mood.mood_score' in data.columns:
            avg_mood = data['mood.mood_score'].mean()
            if avg_mood >= 7:
                strengths.append(f"Positive mood maintenance ({avg_mood:.1f}/10)")
        
        if 'sleep.quality_score' in data.columns:
            avg_quality = data['sleep.quality_score'].mean()
            if avg_quality >= 7:
                strengths.append(f"High sleep quality ({avg_quality:.1f}/10)")
        
        return strengths


# ================================
# CUSTOMIZATION SERVICE
# ================================
class CustomizationService:
    def __init__(self):
        self.themes = self._initialize_themes()
        self.widgets = self._initialize_widgets()
        self.dashboard_templates = self._initialize_dashboard_templates()
    
    def _initialize_themes(self):
        """Initialize theme configurations"""
        return {
            'light': {
                'name': 'Light Theme',
                'background': '#FFFFFF',
                'surface': '#F8F9FA',
                'primary': '#007AFF',
                'secondary': '#5856D6',
                'text_primary': '#000000',
                'text_secondary': '#6C757D',
                'accent': '#34C759',
                'warning': '#FF9500',
                'error': '#FF3B30'
            },
            'dark': {
                'name': 'Dark Theme',
                'background': '#000000',
                'surface': '#1C1C1E',
                'primary': '#0A84FF',
                'secondary': '#5E5CE6',
                'text_primary': '#FFFFFF',
                'text_secondary': '#8E8E93',
                'accent': '#30D158',
                'warning': '#FF9F0A',
                'error': '#FF453A'
            },
            'blue_light': {
                'name': 'Blue Light Filter',
                'background': '#F0F8FF',
                'surface': '#E6F3FF',
                'primary': '#1E90FF',
                'secondary': '#4169E1',
                'text_primary': '#191970',
                'text_secondary': '#4682B4',
                'accent': '#00CED1',
                'warning': '#FF8C00',
                'error': '#DC143C'
            }
        }
    
    def _initialize_widgets(self):
        """Initialize available dashboard widgets"""
        return [
            {
                'id': 'sleep_graph',
                'name': 'Sleep Duration Graph',
                'description': 'Weekly sleep duration trends',
                'category': 'sleep',
                'size': 'large',
                'icon': 'moon'
            },
            {
                'id': 'mood_meter',
                'name': 'Mood Meter',
                'description': 'Current mood and recent trends',
                'category': 'mood',
                'size': 'medium',
                'icon': 'happy'
            },
            {
                'id': 'flare_risk',
                'name': 'Flare Risk Alert',
                'description': 'Predicted symptom flare risk',
                'category': 'symptoms',
                'size': 'small',
                'icon': 'warning'
            },
            {
                'id': 'stress_tracker',
                'name': 'Stress Tracker',
                'description': 'Daily stress levels and patterns',
                'category': 'stress',
                'size': 'medium',
                'icon': 'pulse'
            },
            {
                'id': 'sleep_quality',
                'name': 'Sleep Quality Score',
                'description': 'Sleep quality trends and insights',
                'category': 'sleep',
                'size': 'medium',
                'icon': 'star'
            },
            {
                'id': 'correlation_heatmap',
                'name': 'Health Correlations',
                'description': 'Correlation heatmap of health metrics',
                'category': 'analytics',
                'size': 'large',
                'icon': 'grid'
            },
            {
                'id': 'progress_badges',
                'name': 'Progress Badges',
                'description': 'Achievements and streak tracking',
                'category': 'gamification',
                'size': 'medium',
                'icon': 'trophy'
            },
            {
                'id': 'insights_panel',
                'name': 'Personalized Insights',
                'description': 'AI-generated health insights',
                'category': 'insights',
                'size': 'large',
                'icon': 'bulb'
            }
        ]
    
    def _initialize_dashboard_templates(self):
        """Initialize dashboard layout templates"""
        return [
            {
                'id': 'overview',
                'name': 'Overview Dashboard',
                'description': 'Complete health overview',
                'widgets': ['sleep_graph', 'mood_meter', 'flare_risk', 'stress_tracker', 'progress_badges']
            },
            {
                'id': 'sleep_focused',
                'name': 'Sleep Focused',
                'description': 'Sleep-centric dashboard',
                'widgets': ['sleep_graph', 'sleep_quality', 'mood_meter', 'insights_panel']
            },
            {
                'id': 'analytics',
                'name': 'Analytics Dashboard',
                'description': 'Data-driven insights',
                'widgets': ['correlation_heatmap', 'insights_panel', 'sleep_graph', 'stress_tracker']
            },
            {
                'id': 'minimal',
                'name': 'Minimal Dashboard',
                'description': 'Essential widgets only',
                'widgets': ['mood_meter', 'sleep_quality', 'flare_risk']
            }
        ]
    
    def get_available_widgets(self, category=None):
        """Get available widgets, optionally filtered by category"""
        if category:
            return [widget for widget in self.widgets if widget['category'] == category]
        return self.widgets
    
    def create_custom_dashboard(self, widget_ids, layout='grid'):
        """Create custom dashboard configuration"""
        selected_widgets = [widget for widget in self.widgets if widget['id'] in widget_ids]
        
        return {
            'id': 'custom',
            'name': 'Custom Dashboard',
            'description': 'User-customized layout',
            'widgets': selected_widgets,
            'layout': layout,
            'created_at': datetime.now().isoformat()
        }
    
    def get_theme_config(self, theme_name):
        """Get theme configuration"""
        return self.themes.get(theme_name, self.themes['light'])
    
    def apply_theme(self, theme_name):
        """Apply theme configuration"""
        theme = self.get_theme_config(theme_name)
        return {
            'theme': theme_name,
            'config': theme,
            'applied_at': datetime.now().isoformat()
        }


# ================================
# LONG-TERM VALUE SERVICE
# ================================
class LongTermValueService:
    def __init__(self):
        self.goal_templates = self._initialize_goal_templates()
        self.report_templates = self._initialize_report_templates()
    
    def _initialize_goal_templates(self):
        """Initialize goal setting templates"""
        return [
            {
                'id': 'sleep_duration',
                'name': 'Sleep Duration Goal',
                'description': 'Target average sleep hours per week',
                'category': 'sleep',
                'default_target': 7.5,
                'min_target': 6.0,
                'max_target': 9.0,
                'unit': 'hours',
                'timeframe': 'weekly'
            },
            {
                'id': 'sleep_quality',
                'name': 'Sleep Quality Goal',
                'description': 'Target average sleep quality score',
                'category': 'sleep',
                'default_target': 7.0,
                'min_target': 5.0,
                'max_target': 10.0,
                'unit': 'score',
                'timeframe': 'weekly'
            },
            {
                'id': 'stress_reduction',
                'name': 'Stress Reduction Goal',
                'description': 'Target maximum stress level',
                'category': 'stress',
                'default_target': 6.0,
                'min_target': 3.0,
                'max_target': 8.0,
                'unit': 'score',
                'timeframe': 'weekly'
            },
            {
                'id': 'mood_improvement',
                'name': 'Mood Improvement Goal',
                'description': 'Target minimum mood score',
                'category': 'mood',
                'default_target': 7.0,
                'min_target': 5.0,
                'max_target': 10.0,
                'unit': 'score',
                'timeframe': 'weekly'
            },
            {
                'id': 'consistency',
                'name': 'Logging Consistency Goal',
                'description': 'Target days logged per week',
                'category': 'habits',
                'default_target': 7.0,
                'min_target': 3.0,
                'max_target': 7.0,
                'unit': 'days',
                'timeframe': 'weekly'
            }
        ]
    
    def _initialize_report_templates(self):
        """Initialize report templates"""
        return {
            'weekly_summary': {
                'name': 'Weekly Summary',
                'timeframe': '7_days',
                'sections': ['highlights', 'trends', 'goals', 'insights']
            },
            'monthly_report': {
                'name': 'Monthly Report',
                'timeframe': '30_days',
                'sections': ['overview', 'achievements', 'trends', 'correlations', 'recommendations']
            },
            'quarterly_review': {
                'name': 'Quarterly Review',
                'timeframe': '90_days',
                'sections': ['progress', 'milestones', 'patterns', 'goals', 'future_planning']
            }
        }
    
    def create_goal(self, goal_type, target_value, timeframe='weekly'):
        """Create a new goal"""
        template = next((g for g in self.goal_templates if g['id'] == goal_type), None)
        if not template:
            return None
        
        # Validate target value
        if target_value < template['min_target'] or target_value > template['max_target']:
            return None
        
        return {
            'id': f"{goal_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'type': goal_type,
            'name': template['name'],
            'description': template['description'],
            'target_value': target_value,
            'current_value': 0,
            'progress_percentage': 0,
            'timeframe': timeframe,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'category': template['category'],
            'unit': template['unit']
        }
    
    def update_goal_progress(self, goal, user_data):
        """Update goal progress based on user data"""
        if goal['type'] == 'sleep_duration':
            current_value = user_data['sleep.duration_hours'].mean() if 'sleep.duration_hours' in user_data.columns else 0
        elif goal['type'] == 'sleep_quality':
            current_value = user_data['sleep.quality_score'].mean() if 'sleep.quality_score' in user_data.columns else 0
        elif goal['type'] == 'stress_reduction':
            current_value = user_data['mood.stress_score'].mean() if 'mood.stress_score' in user_data.columns else 0
        elif goal['type'] == 'mood_improvement':
            current_value = user_data['mood.mood_score'].mean() if 'mood.mood_score' in user_data.columns else 0
        elif goal['type'] == 'consistency':
            current_value = len(user_data)  # Days logged
        else:
            return goal
        
        goal['current_value'] = current_value
        
        # Calculate progress percentage
        if goal['type'] == 'stress_reduction':
            # For stress reduction, lower is better
            progress = max(0, min(100, (goal['target_value'] - current_value) / goal['target_value'] * 100))
        else:
            # For other goals, higher is better
            progress = max(0, min(100, current_value / goal['target_value'] * 100))
        
        goal['progress_percentage'] = progress
        
        # Update status
        if progress >= 100:
            goal['status'] = 'achieved'
        elif progress >= 80:
            goal['status'] = 'on_track'
        elif progress >= 50:
            goal['status'] = 'needs_attention'
        else:
            goal['status'] = 'at_risk'
        
        return goal
    
    def generate_personalized_report(self, user_data, report_type, time_range='90_days'):
        """Generate personalized long-term report"""
        if report_type not in self.report_templates:
            return None
        
        template = self.report_templates[report_type]
        
        # Filter data by time range
        filtered_data = self._filter_data_by_time_range(user_data, time_range)
        
        if len(filtered_data) == 0:
            return None
        
        report = {
            'type': report_type,
            'name': template['name'],
            'generated_at': datetime.now().isoformat(),
            'time_range': time_range,
            'data_period': f"{filtered_data.index[0].strftime('%Y-%m-%d')} to {filtered_data.index[-1].strftime('%Y-%m-%d')}",
            'sections': {}
        }
        
        # Generate each section
        for section in template['sections']:
            report['sections'][section] = self._generate_report_section(filtered_data, section, time_range)
        
        return report
    
    def _filter_data_by_time_range(self, data, time_range):
        """Filter data by time range"""
        now = pd.Timestamp.now()
        
        if time_range == '7_days':
            cutoff_date = now - pd.Timedelta(days=7)
        elif time_range == '30_days':
            cutoff_date = now - pd.Timedelta(days=30)
        elif time_range == '90_days':
            cutoff_date = now - pd.Timedelta(days=90)
        elif time_range == '180_days':
            cutoff_date = now - pd.Timedelta(days=180)
        else:
            return data
        
        # Handle both indexed and non-indexed DataFrames
        if 'date' in data.columns:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])
            return data_copy[data_copy['date'] >= cutoff_date]
        else:
            return data[data.index >= cutoff_date]
    
    def _generate_report_section(self, data, section, time_range):
        """Generate specific report section"""
        if section == 'highlights':
            return self._generate_highlights(data)
        elif section == 'trends':
            return self._generate_trends(data)
        elif section == 'goals':
            return self._generate_goals_progress(data)
        elif section == 'insights':
            return self._generate_insights(data)
        elif section == 'overview':
            return self._generate_overview(data)
        elif section == 'achievements':
            return self._generate_achievements(data)
        elif section == 'correlations':
            return self._generate_correlations(data)
        elif section == 'recommendations':
            return self._generate_recommendations(data)
        else:
            return {}
    
    def _generate_highlights(self, data):
        """Generate weekly highlights"""
        highlights = []
        
        # Sleep highlights
        if 'sleep.duration_hours' in data.columns:
            avg_sleep = data['sleep.duration_hours'].mean()
            best_sleep = data['sleep.duration_hours'].max()
            highlights.append(f"Average sleep: {avg_sleep:.1f} hours (best night: {best_sleep:.1f}h)")
        
        # Mood highlights
        if 'mood.mood_score' in data.columns:
            avg_mood = data['mood.mood_score'].mean()
            best_mood = data['mood.mood_score'].max()
            highlights.append(f"Average mood: {avg_mood:.1f}/10 (best day: {best_mood:.1f}/10)")
        
        # Consistency highlight
        days_logged = len(data)
        highlights.append(f"Logged {days_logged} days this week")
        
        return highlights
    
    def _generate_trends(self, data):
        """Generate trend analysis"""
        trends = []
        
        # Sleep trend
        if 'sleep.duration_hours' in data.columns and len(data) > 3:
            sleep_trend = self._calculate_trend_direction(data['sleep.duration_hours'])
            trends.append(f"Sleep duration: {sleep_trend}")
        
        # Mood trend
        if 'mood.mood_score' in data.columns and len(data) > 3:
            mood_trend = self._calculate_trend_direction(data['mood.mood_score'])
            trends.append(f"Mood: {mood_trend}")
        
        # Stress trend
        if 'mood.stress_score' in data.columns and len(data) > 3:
            stress_trend = self._calculate_trend_direction(data['mood.stress_score'])
            trends.append(f"Stress: {stress_trend}")
        
        return trends
    
    def _calculate_trend_direction(self, series):
        """Calculate trend direction for a series"""
        if len(series) < 3:
            return "insufficient data"
        
        # Simple linear regression
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _generate_goals_progress(self, data):
        """Generate goals progress summary"""
        # This would integrate with actual goals in a real implementation
        return {
            'active_goals': 3,
            'achieved_goals': 1,
            'on_track_goals': 2,
            'at_risk_goals': 0
        }
    
    def _generate_insights(self, data):
        """Generate AI insights"""
        insights = []
        
        # Sleep insights
        if 'sleep.duration_hours' in data.columns and 'sleep.quality_score' in data.columns:
            sleep_duration = data['sleep.duration_hours'].mean()
            sleep_quality = data['sleep.quality_score'].mean()
            
            if sleep_duration >= 7.5 and sleep_quality >= 7:
                insights.append("Excellent sleep habits! You're getting both sufficient duration and high quality sleep.")
            elif sleep_duration < 7:
                insights.append("Consider increasing your sleep duration. Most adults need 7-9 hours for optimal health.")
        
        # Mood insights
        if 'mood.mood_score' in data.columns and 'mood.stress_score' in data.columns:
            mood = data['mood.mood_score'].mean()
            stress = data['mood.stress_score'].mean()
            
            if mood >= 7 and stress <= 5:
                insights.append("Great mental health balance! Your mood is positive and stress is well-managed.")
            elif stress > 7:
                insights.append("High stress levels detected. Consider stress management techniques like meditation or exercise.")
        
        return insights
    
    def _generate_overview(self, data):
        """Generate overview statistics"""
        overview = {
            'total_days': len(data),
            'data_completeness': self._calculate_data_completeness(data),
            'key_metrics': {}
        }
        
        # Key metrics
        if 'sleep.duration_hours' in data.columns:
            overview['key_metrics']['avg_sleep_duration'] = data['sleep.duration_hours'].mean()
        if 'mood.mood_score' in data.columns:
            overview['key_metrics']['avg_mood'] = data['mood.mood_score'].mean()
        if 'mood.stress_score' in data.columns:
            overview['key_metrics']['avg_stress'] = data['mood.stress_score'].mean()
        
        return overview
    
    def _calculate_data_completeness(self, data):
        """Calculate data completeness percentage"""
        total_fields = len(data.columns)
        complete_fields = 0
        
        for column in data.columns:
            if data[column].notna().sum() > 0:
                complete_fields += 1
        
        return (complete_fields / total_fields) * 100 if total_fields > 0 else 0
    
    def _generate_achievements(self, data):
        """Generate achievements summary"""
        achievements = []
        
        # Consistency achievement
        days_logged = len(data)
        if days_logged >= 7:
            achievements.append("7-day logging streak!")
        elif days_logged >= 30:
            achievements.append("30-day logging streak!")
        
        # Sleep achievements
        if 'sleep.duration_hours' in data.columns:
            avg_sleep = data['sleep.duration_hours'].mean()
            if avg_sleep >= 8:
                achievements.append("Sleep champion (8+ hours average)!")
        
        return achievements
    
    def _generate_correlations(self, data):
        """Generate correlation insights"""
        correlations = []
        
        # Sleep-mood correlation
        if 'sleep.duration_hours' in data.columns and 'mood.mood_score' in data.columns:
            corr = data['sleep.duration_hours'].corr(data['mood.mood_score'])
            if abs(corr) > 0.5:
                direction = "positive" if corr > 0 else "negative"
                correlations.append(f"Strong {direction} correlation between sleep duration and mood")
        
        return correlations
    
    def _generate_recommendations(self, data):
        """Generate personalized recommendations"""
        recommendations = []
        
        # Sleep recommendations
        if 'sleep.duration_hours' in data.columns:
            avg_sleep = data['sleep.duration_hours'].mean()
            if avg_sleep < 7:
                recommendations.append("Aim for 7-9 hours of sleep nightly for better health and mood")
        
        # Stress recommendations
        if 'mood.stress_score' in data.columns:
            avg_stress = data['mood.stress_score'].mean()
            if avg_stress > 7:
                recommendations.append("Consider stress management techniques like deep breathing or meditation")
        
        return recommendations


# ================================
# CROSS-APP INTEGRATION SERVICE
# ================================
class CrossAppIntegrationService:
    def __init__(self):
        self.app_configs = self._initialize_app_configs()
        self.integration_templates = self._initialize_integration_templates()
        self.data_mappings = self._initialize_data_mappings()
    
    def _initialize_app_configs(self):
        """Initialize configurations for integrated apps"""
        return {
            'mindmap': {
                'name': 'MindMap',
                'description': 'Mental health and cognitive tracking',
                'data_types': ['mood', 'anxiety', 'depression', 'cognitive_tests', 'meditation'],
                'api_endpoint': 'mindmap://api/v1',
                'sync_frequency': 'daily',
                'priority': 'high'
            },
            'skintrack': {
                'name': 'SkinTrack+',
                'description': 'Skin condition and flare tracking',
                'data_types': ['skin_flares', 'dermatitis', 'psoriasis', 'eczema', 'treatments'],
                'api_endpoint': 'skintrack://api/v1',
                'sync_frequency': 'daily',
                'priority': 'high'
            },
            'gastroguard': {
                'name': 'GastroGuard',
                'description': 'Digestive health and GI tracking',
                'data_types': ['gi_flares', 'food_intake', 'digestive_symptoms', 'medications'],
                'api_endpoint': 'gastroguard://api/v1',
                'sync_frequency': 'daily',
                'priority': 'high'
            }
        }
    
    def _initialize_integration_templates(self):
        """Initialize integration templates for cross-app data sharing"""
        return {
            'unified_health_dashboard': {
                'name': 'Unified Health Dashboard',
                'description': 'Comprehensive view across all health apps',
                'widgets': ['sleep_insights', 'mood_correlations', 'symptom_patterns', 'treatment_effects'],
                'refresh_rate': 'real_time'
            },
            'cross_app_correlations': {
                'name': 'Cross-App Correlations',
                'description': 'Find patterns across different health metrics',
                'analyses': ['sleep_mood_gi', 'stress_skin_flares', 'meditation_symptom_relief'],
                'update_frequency': 'weekly'
            },
            'unified_insights': {
                'name': 'Unified Health Insights',
                'description': 'AI insights combining data from all apps',
                'sections': ['holistic_health', 'treatment_optimization', 'lifestyle_recommendations'],
                'generation_frequency': 'daily'
            }
        }
    
    def _initialize_data_mappings(self):
        """Initialize data field mappings between apps"""
        return {
            'sleep_to_mindmap': {
                'sleep.duration_hours': 'sleep_quality_score',
                'sleep.quality_score': 'rest_effectiveness',
                'mood.mood_score': 'daily_mood',
                'mood.stress_score': 'stress_level'
            },
            'sleep_to_skintrack': {
                'sleep.quality_score': 'skin_health_factor',
                'mood.stress_score': 'stress_trigger',
                'symptoms.skin_flare': 'flare_severity'
            },
            'sleep_to_gastroguard': {
                'sleep.duration_hours': 'digestive_health_factor',
                'mood.stress_score': 'gi_stress_trigger',
                'symptoms.gi_flare': 'gi_symptom_severity'
            },
            'mindmap_to_sleep': {
                'anxiety_score': 'sleep_disruption_factor',
                'meditation_minutes': 'sleep_quality_boost',
                'cognitive_fatigue': 'sleep_need_indicator'
            },
            'skintrack_to_sleep': {
                'flare_severity': 'sleep_disruption',
                'itch_intensity': 'sleep_quality_impact',
                'treatment_effectiveness': 'sleep_improvement'
            },
            'gastroguard_to_sleep': {
                'gi_symptom_severity': 'sleep_disruption',
                'food_sensitivity': 'sleep_quality_impact',
                'digestive_comfort': 'sleep_quality_factor'
            }
        }
    
    def sync_app_data(self, app_name, data):
        """Sync data with specified app"""
        if app_name not in self.app_configs:
            return None
        
        app_config = self.app_configs[app_name]
        
        # Simulate API call to app
        sync_result = {
            'app': app_name,
            'sync_timestamp': datetime.now().isoformat(),
            'data_points_synced': len(data),
            'status': 'success',
            'mapped_fields': self._map_data_fields(data, app_name)
        }
        
        return sync_result
    
    def _map_data_fields(self, data, target_app):
        """Map data fields to target app format"""
        mapping_key = f'sleep_to_{target_app}'
        if mapping_key not in self.data_mappings:
            return {}
        
        mapping = self.data_mappings[mapping_key]
        mapped_data = {}
        
        for source_field, target_field in mapping.items():
            if source_field in data.columns:
                mapped_data[target_field] = data[source_field].iloc[-1] if len(data) > 0 else 0
        
        return mapped_data
    
    def generate_unified_insights(self, sleep_data, mindmap_data=None, skintrack_data=None, gastroguard_data=None):
        """Generate insights combining data from all apps"""
        insights = []
        
        # Cross-app correlation analysis
        if mindmap_data is not None:
            insights.extend(self._analyze_sleep_mindmap_correlations(sleep_data, mindmap_data))
        
        if skintrack_data is not None:
            insights.extend(self._analyze_sleep_skin_correlations(sleep_data, skintrack_data))
        
        if gastroguard_data is not None:
            insights.extend(self._analyze_sleep_gi_correlations(sleep_data, gastroguard_data))
        
        # Holistic health insights
        insights.extend(self._generate_holistic_insights(sleep_data, mindmap_data, skintrack_data, gastroguard_data))
        
        return insights
    
    def _analyze_sleep_mindmap_correlations(self, sleep_data, mindmap_data):
        """Analyze correlations between sleep and mental health data"""
        insights = []
        
        # Sleep quality vs anxiety correlation
        if 'sleep.quality_score' in sleep_data.columns and 'anxiety_score' in mindmap_data.columns:
            sleep_quality = sleep_data['sleep.quality_score'].mean()
            anxiety_level = mindmap_data['anxiety_score'].mean()
            
            if sleep_quality < 6 and anxiety_level > 7:
                insights.append({
                    'type': 'correlation',
                    'category': 'sleep_mindmap',
                    'title': 'Sleep-Anxiety Connection',
                    'description': f'Poor sleep quality ({sleep_quality:.1f}/10) correlates with high anxiety ({anxiety_level:.1f}/10)',
                    'recommendation': 'Consider sleep hygiene improvements to help manage anxiety',
                    'priority': 'high'
                })
        
        # Meditation impact on sleep
        if 'meditation_minutes' in mindmap_data.columns and 'sleep.quality_score' in sleep_data.columns:
            meditation_avg = mindmap_data['meditation_minutes'].mean()
            sleep_quality = sleep_data['sleep.quality_score'].mean()
            
            if meditation_avg > 10 and sleep_quality > 7:
                insights.append({
                    'type': 'positive_correlation',
                    'category': 'sleep_mindmap',
                    'title': 'Meditation-Sleep Synergy',
                    'description': f'Regular meditation ({meditation_avg:.0f} min/day) is associated with better sleep quality ({sleep_quality:.1f}/10)',
                    'recommendation': 'Continue meditation practice for optimal sleep',
                    'priority': 'medium'
                })
        
        return insights
    
    def _analyze_sleep_skin_correlations(self, sleep_data, skintrack_data):
        """Analyze correlations between sleep and skin health"""
        insights = []
        
        # Sleep quality vs skin flare correlation
        if 'sleep.quality_score' in sleep_data.columns and 'flare_severity' in skintrack_data.columns:
            sleep_quality = sleep_data['sleep.quality_score'].mean()
            flare_severity = skintrack_data['flare_severity'].mean()
            
            if sleep_quality < 6 and flare_severity > 6:
                insights.append({
                    'type': 'correlation',
                    'category': 'sleep_skintrack',
                    'title': 'Sleep-Skin Health Connection',
                    'description': f'Poor sleep quality ({sleep_quality:.1f}/10) may worsen skin flares ({flare_severity:.1f}/10)',
                    'recommendation': 'Prioritize sleep quality to help manage skin conditions',
                    'priority': 'high'
                })
        
        # Stress-skin flare correlation
        if 'mood.stress_score' in sleep_data.columns and 'flare_severity' in skintrack_data.columns:
            stress_level = sleep_data['mood.stress_score'].mean()
            flare_severity = skintrack_data['flare_severity'].mean()
            
            if stress_level > 7 and flare_severity > 5:
                insights.append({
                    'type': 'correlation',
                    'category': 'sleep_skintrack',
                    'title': 'Stress-Skin Flare Connection',
                    'description': f'High stress ({stress_level:.1f}/10) correlates with increased skin flares ({flare_severity:.1f}/10)',
                    'recommendation': 'Stress management techniques may help reduce skin flare frequency',
                    'priority': 'high'
                })
        
        return insights
    
    def _analyze_sleep_gi_correlations(self, sleep_data, gastroguard_data):
        """Analyze correlations between sleep and digestive health"""
        insights = []
        
        # Sleep duration vs GI symptoms
        if 'sleep.duration_hours' in sleep_data.columns and 'gi_symptom_severity' in gastroguard_data.columns:
            sleep_duration = sleep_data['sleep.duration_hours'].mean()
            gi_severity = gastroguard_data['gi_symptom_severity'].mean()
            
            if sleep_duration < 7 and gi_severity > 5:
                insights.append({
                    'type': 'correlation',
                    'category': 'sleep_gastroguard',
                    'title': 'Sleep-Digestive Health Connection',
                    'description': f'Insufficient sleep ({sleep_duration:.1f}h) may worsen GI symptoms ({gi_severity:.1f}/10)',
                    'recommendation': 'Aim for 7-9 hours of sleep to support digestive health',
                    'priority': 'high'
                })
        
        # Food sensitivity impact on sleep
        if 'food_sensitivity_score' in gastroguard_data.columns and 'sleep.quality_score' in sleep_data.columns:
            sensitivity = gastroguard_data['food_sensitivity_score'].mean()
            sleep_quality = sleep_data['sleep.quality_score'].mean()
            
            if sensitivity > 6 and sleep_quality < 6:
                insights.append({
                    'type': 'correlation',
                    'category': 'sleep_gastroguard',
                    'title': 'Food Sensitivity-Sleep Connection',
                    'description': f'High food sensitivity ({sensitivity:.1f}/10) may impact sleep quality ({sleep_quality:.1f}/10)',
                    'recommendation': 'Consider dietary adjustments to improve sleep quality',
                    'priority': 'medium'
                })
        
        return insights
    
    def _generate_holistic_insights(self, sleep_data, mindmap_data, skintrack_data, gastroguard_data):
        """Generate holistic health insights combining all app data"""
        insights = []
        
        # Overall health score calculation
        health_scores = {}
        
        if sleep_data is not None:
            health_scores['sleep'] = sleep_data['sleep.quality_score'].mean() if 'sleep.quality_score' in sleep_data.columns else 0
        
        if mindmap_data is not None:
            health_scores['mental'] = mindmap_data['mood_score'].mean() if 'mood_score' in mindmap_data.columns else 0
        
        if skintrack_data is not None:
            health_scores['skin'] = 10 - skintrack_data['flare_severity'].mean() if 'flare_severity' in skintrack_data.columns else 0
        
        if gastroguard_data is not None:
            health_scores['digestive'] = 10 - gastroguard_data['gi_symptom_severity'].mean() if 'gi_symptom_severity' in gastroguard_data.columns else 0
        
        # Calculate overall health score
        if health_scores:
            overall_score = sum(health_scores.values()) / len(health_scores)
            
            if overall_score >= 8:
                insights.append({
                    'type': 'holistic',
                    'category': 'overall_health',
                    'title': 'Excellent Holistic Health',
                    'description': f'Your overall health score is {overall_score:.1f}/10 across all tracked areas',
                    'recommendation': 'Continue your current health practices',
                    'priority': 'low'
                })
            elif overall_score < 6:
                insights.append({
                    'type': 'holistic',
                    'category': 'overall_health',
                    'title': 'Holistic Health Improvement Needed',
                    'description': f'Your overall health score is {overall_score:.1f}/10 - focus on the lowest scoring areas',
                    'recommendation': 'Prioritize improvements in the areas with lowest scores',
                    'priority': 'high'
                })
        
        # Treatment optimization insights
        if mindmap_data is not None and 'medication_effectiveness' in mindmap_data.columns:
            med_effectiveness = mindmap_data['medication_effectiveness'].mean()
            if med_effectiveness < 6:
                insights.append({
                    'type': 'treatment',
                    'category': 'medication_optimization',
                    'title': 'Medication Effectiveness Review',
                    'description': f'Current medication effectiveness is {med_effectiveness:.1f}/10',
                    'recommendation': 'Consider discussing medication adjustments with your healthcare provider',
                    'priority': 'medium'
                })
        
        return insights
    
    def generate_unified_dashboard_data(self, sleep_data, mindmap_data=None, skintrack_data=None, gastroguard_data=None):
        """Generate data for unified health dashboard"""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'apps_connected': [],
            'metrics': {},
            'correlations': {},
            'insights': []
        }
        
        # Track connected apps
        if sleep_data is not None:
            dashboard_data['apps_connected'].append('Sleep Tracker')
        if mindmap_data is not None:
            dashboard_data['apps_connected'].append('MindMap')
        if skintrack_data is not None:
            dashboard_data['apps_connected'].append('SkinTrack+')
        if gastroguard_data is not None:
            dashboard_data['apps_connected'].append('GastroGuard')
        
        # Aggregate metrics
        if sleep_data is not None:
            dashboard_data['metrics']['sleep_quality'] = sleep_data['sleep.quality_score'].mean() if 'sleep.quality_score' in sleep_data.columns else 0
            dashboard_data['metrics']['sleep_duration'] = sleep_data['sleep.duration_hours'].mean() if 'sleep.duration_hours' in sleep_data.columns else 0
            dashboard_data['metrics']['stress_level'] = sleep_data['mood.stress_score'].mean() if 'mood.stress_score' in sleep_data.columns else 0
        
        if mindmap_data is not None:
            dashboard_data['metrics']['anxiety_level'] = mindmap_data['anxiety_score'].mean() if 'anxiety_score' in mindmap_data.columns else 0
            dashboard_data['metrics']['mood_score'] = mindmap_data['mood_score'].mean() if 'mood_score' in mindmap_data.columns else 0
        
        if skintrack_data is not None:
            dashboard_data['metrics']['skin_flare_severity'] = skintrack_data['flare_severity'].mean() if 'flare_severity' in skintrack_data.columns else 0
        
        if gastroguard_data is not None:
            dashboard_data['metrics']['gi_symptom_severity'] = gastroguard_data['gi_symptom_severity'].mean() if 'gi_symptom_severity' in gastroguard_data.columns else 0
        
        # Generate unified insights
        dashboard_data['insights'] = self.generate_unified_insights(sleep_data, mindmap_data, skintrack_data, gastroguard_data)
        
        return dashboard_data


# ================================
# USER PROFILE SERVICE
# ================================
class UserProfileService:
    def __init__(self):
        self.profile_data = {}
        self.health_conditions = self._initialize_health_conditions()
        self.bmi_categories = self._initialize_bmi_categories()
        self.age_groups = self._initialize_age_groups()
    
    def _initialize_health_conditions(self):
        """Initialize common health conditions for tracking"""
        return {
            'mental_health': [
                'anxiety', 'depression', 'bipolar_disorder', 'ptsd', 'adhd', 
                'ocd', 'eating_disorders', 'substance_use_disorder'
            ],
            'physical_health': [
                'diabetes', 'hypertension', 'heart_disease', 'asthma', 'copd',
                'arthritis', 'fibromyalgia', 'chronic_fatigue', 'autoimmune_disorders'
            ],
            'digestive_health': [
                'ibs', 'crohns_disease', 'ulcerative_colitis', 'celiac_disease',
                'gerd', 'gastritis', 'food_intolerances', 'lactose_intolerance'
            ],
            'skin_conditions': [
                'eczema', 'psoriasis', 'dermatitis', 'acne', 'rosacea',
                'vitiligo', 'alopecia', 'chronic_urticaria'
            ],
            'sleep_disorders': [
                'insomnia', 'sleep_apnea', 'restless_leg_syndrome', 'narcolepsy',
                'circadian_rhythm_disorders', 'night_terrors', 'sleep_walking'
            ],
            'neurological': [
                'migraine', 'epilepsy', 'multiple_sclerosis', 'parkinsons',
                'alzheimers', 'chronic_pain', 'neuropathy'
            ]
        }
    
    def _initialize_bmi_categories(self):
        """Initialize BMI categories for health assessment"""
        return {
            'underweight': {'min': 0, 'max': 18.5, 'description': 'Underweight'},
            'normal': {'min': 18.5, 'max': 25, 'description': 'Normal weight'},
            'overweight': {'min': 25, 'max': 30, 'description': 'Overweight'},
            'obese_class_1': {'min': 30, 'max': 35, 'description': 'Obese Class I'},
            'obese_class_2': {'min': 35, 'max': 40, 'description': 'Obese Class II'},
            'obese_class_3': {'min': 40, 'max': 100, 'description': 'Obese Class III'}
        }
    
    def _initialize_age_groups(self):
        """Initialize age groups for health recommendations"""
        return {
            'adolescent': {'min': 13, 'max': 17, 'description': 'Adolescent (13-17)'},
            'young_adult': {'min': 18, 'max': 25, 'description': 'Young Adult (18-25)'},
            'adult': {'min': 26, 'max': 40, 'description': 'Adult (26-40)'},
            'middle_aged': {'min': 41, 'max': 60, 'description': 'Middle-aged (41-60)'},
            'senior': {'min': 61, 'max': 100, 'description': 'Senior (61+)'}
        }
    
    def create_user_profile(self, profile_data):
        """Create or update user profile with health information"""
        # Validate required fields
        required_fields = ['age', 'gender', 'height_cm', 'weight_kg']
        for field in required_fields:
            if field not in profile_data:
                return {'success': False, 'error': f'Missing required field: {field}'}
        
        # Validate data types and ranges
        validation_result = self._validate_profile_data(profile_data)
        if not validation_result['valid']:
            return {'success': False, 'error': validation_result['error']}
        
        # Calculate derived metrics
        profile_data['bmi'] = self._calculate_bmi(profile_data['height_cm'], profile_data['weight_kg'])
        profile_data['bmi_category'] = self._get_bmi_category(profile_data['bmi'])
        profile_data['age_group'] = self._get_age_group(profile_data['age'])
        
        # Add metadata
        profile_data['created_at'] = datetime.now().isoformat()
        profile_data['last_updated'] = datetime.now().isoformat()
        profile_data['profile_completeness'] = self._calculate_profile_completeness(profile_data)
        
        # Store profile
        self.profile_data = profile_data
        
        return {
            'success': True,
            'profile': profile_data,
            'insights': self._generate_profile_insights(profile_data)
        }
    
    def _validate_profile_data(self, data):
        """Validate profile data for correctness"""
        # Age validation
        if not isinstance(data['age'], (int, float)) or data['age'] < 13 or data['age'] > 120:
            return {'valid': False, 'error': 'Age must be between 13 and 120'}
        
        # Gender validation
        valid_genders = ['male', 'female', 'nonbinary', 'trans', 'other']
        if data['gender'].lower() not in valid_genders:
            return {'valid': False, 'error': f'Gender must be one of: {", ".join(valid_genders)}'}
        
        # Height validation (in cm)
        if not isinstance(data['height_cm'], (int, float)) or data['height_cm'] < 100 or data['height_cm'] > 250:
            return {'valid': False, 'error': 'Height must be between 100-250 cm'}
        
        # Weight validation (in kg)
        if not isinstance(data['weight_kg'], (int, float)) or data['weight_kg'] < 20 or data['weight_kg'] > 300:
            return {'valid': False, 'error': 'Weight must be between 20-300 kg'}
        
        # Optional fields validation
        if 'known_conditions' in data:
            if not isinstance(data['known_conditions'], list):
                return {'valid': False, 'error': 'Known conditions must be a list'}
        
        return {'valid': True}
    
    def _calculate_bmi(self, height_cm, weight_kg):
        """Calculate BMI from height and weight"""
        height_m = height_cm / 100
        return round(weight_kg / (height_m ** 2), 1)
    
    def _get_bmi_category(self, bmi):
        """Get BMI category based on BMI value"""
        for category, range_data in self.bmi_categories.items():
            if range_data['min'] <= bmi < range_data['max']:
                return {
                    'category': category,
                    'description': range_data['description'],
                    'bmi': bmi
                }
        return {'category': 'unknown', 'description': 'Unknown', 'bmi': bmi}
    
    def _get_age_group(self, age):
        """Get age group based on age"""
        for group, range_data in self.age_groups.items():
            if range_data['min'] <= age <= range_data['max']:
                return {
                    'group': group,
                    'description': range_data['description'],
                    'age': age
                }
        return {'group': 'unknown', 'description': 'Unknown', 'age': age}
    
    def _calculate_profile_completeness(self, profile_data):
        """Calculate profile completeness percentage"""
        total_fields = 8  # age, gender, height, weight, conditions, medications, allergies, activity_level
        completed_fields = 0
        
        required_fields = ['age', 'gender', 'height_cm', 'weight_kg']
        for field in required_fields:
            if field in profile_data and profile_data[field] is not None:
                completed_fields += 1
        
        optional_fields = ['known_conditions', 'medications', 'allergies', 'activity_level']
        for field in optional_fields:
            if field in profile_data and profile_data[field] is not None:
                completed_fields += 1
        
        return round((completed_fields / total_fields) * 100, 1)
    
    def _generate_profile_insights(self, profile_data):
        """Generate personalized insights based on profile data"""
        insights = []
        
        # BMI insights
        bmi_info = profile_data['bmi_category']
        if bmi_info['category'] == 'underweight':
            insights.append({
                'type': 'health',
                'category': 'nutrition',
                'title': 'Underweight BMI',
                'description': f'Your BMI is {bmi_info["bmi"]} ({bmi_info["description"]})',
                'recommendation': 'Consider consulting a healthcare provider about healthy weight gain strategies',
                'priority': 'medium'
            })
        elif bmi_info['category'] in ['overweight', 'obese_class_1', 'obese_class_2', 'obese_class_3']:
            insights.append({
                'type': 'health',
                'category': 'nutrition',
                'title': 'Weight Management',
                'description': f'Your BMI is {bmi_info["bmi"]} ({bmi_info["description"]})',
                'recommendation': 'Consider lifestyle modifications for healthy weight management',
                'priority': 'medium'
            })
        
        # Age-specific insights
        age_info = profile_data['age_group']
        if age_info['group'] == 'senior':
            insights.append({
                'type': 'health',
                'category': 'age_related',
                'title': 'Senior Health Considerations',
                'description': f'You are in the {age_info["description"]} group',
                'recommendation': 'Regular health checkups and monitoring become increasingly important',
                'priority': 'low'
            })
        elif age_info['group'] == 'adolescent':
            insights.append({
                'type': 'health',
                'category': 'age_related',
                'title': 'Adolescent Health',
                'description': f'You are in the {age_info["description"]} group',
                'recommendation': 'Focus on establishing healthy sleep and stress management habits early',
                'priority': 'low'
            })
        
        # Condition-specific insights
        if 'known_conditions' in profile_data and profile_data['known_conditions']:
            conditions = profile_data['known_conditions']
            
            # Mental health conditions
            mental_conditions = [c for c in conditions if c in self.health_conditions['mental_health']]
            if mental_conditions:
                insights.append({
                    'type': 'health',
                    'category': 'mental_health',
                    'title': 'Mental Health Support',
                    'description': f'You have {len(mental_conditions)} mental health condition(s) tracked',
                    'recommendation': 'Regular monitoring of mood, stress, and sleep patterns is especially important',
                    'priority': 'high'
                })
            
            # Sleep disorders
            sleep_conditions = [c for c in conditions if c in self.health_conditions['sleep_disorders']]
            if sleep_conditions:
                insights.append({
                    'type': 'health',
                    'category': 'sleep',
                    'title': 'Sleep Disorder Management',
                    'description': f'You have {len(sleep_conditions)} sleep disorder(s)',
                    'recommendation': 'Detailed sleep tracking and consultation with a sleep specialist may be beneficial',
                    'priority': 'high'
                })
            
            # Digestive conditions
            digestive_conditions = [c for c in conditions if c in self.health_conditions['digestive_health']]
            if digestive_conditions:
                insights.append({
                    'type': 'health',
                    'category': 'digestive',
                    'title': 'Digestive Health Monitoring',
                    'description': f'You have {len(digestive_conditions)} digestive condition(s)',
                    'recommendation': 'Track food intake, symptoms, and sleep patterns for better management',
                    'priority': 'high'
                })
        
        return insights
    
    def get_personalized_recommendations(self, profile_data, health_data=None):
        """Get personalized recommendations based on profile and health data"""
        recommendations = []
        
        # Age-based recommendations
        age = profile_data['age']
        if age >= 65:
            recommendations.append({
                'category': 'sleep',
                'title': 'Senior Sleep Patterns',
                'description': 'Older adults may need 7-8 hours of sleep, with more frequent awakenings being normal',
                'action': 'Track sleep quality rather than just duration'
            })
        elif age <= 25:
            recommendations.append({
                'category': 'sleep',
                'title': 'Young Adult Sleep',
                'description': 'Young adults often have delayed sleep phases and may need 7-9 hours',
                'action': 'Maintain consistent sleep schedule even on weekends'
            })
        
        # Gender-based recommendations
        gender = profile_data['gender'].lower()
        if gender == 'female':
            recommendations.append({
                'category': 'health',
                'title': 'Hormonal Health',
                'description': 'Hormonal fluctuations can affect sleep, mood, and stress levels',
                'action': 'Track menstrual cycle patterns if applicable'
            })
        
        # BMI-based recommendations
        bmi_category = profile_data['bmi_category']['category']
        if bmi_category in ['overweight', 'obese_class_1', 'obese_class_2', 'obese_class_3']:
            recommendations.append({
                'category': 'sleep',
                'title': 'Weight and Sleep',
                'description': 'Excess weight can contribute to sleep apnea and poor sleep quality',
                'action': 'Monitor for sleep breathing issues and consider weight management'
            })
        
        # Condition-specific recommendations
        if 'known_conditions' in profile_data and profile_data['known_conditions']:
            conditions = profile_data['known_conditions']
            
            if 'anxiety' in conditions or 'depression' in conditions:
                recommendations.append({
                    'category': 'mental_health',
                    'title': 'Mental Health and Sleep',
                    'description': 'Mental health conditions often impact sleep quality and duration',
                    'action': 'Prioritize sleep hygiene and consider therapy or medication if needed'
                })
            
            if 'diabetes' in conditions:
                recommendations.append({
                    'category': 'health',
                    'title': 'Diabetes and Sleep',
                    'description': 'Blood sugar levels can affect sleep quality and vice versa',
                    'action': 'Monitor blood sugar patterns in relation to sleep and stress'
                })
        
        return recommendations
    
    def update_profile(self, updates):
        """Update specific profile fields"""
        if not self.profile_data:
            return {'success': False, 'error': 'No profile exists to update'}
        
        # Update fields
        for key, value in updates.items():
            if key in ['age', 'height_cm', 'weight_kg']:
                self.profile_data[key] = value
            elif key == 'known_conditions':
                self.profile_data[key] = value
            elif key in ['medications', 'allergies', 'activity_level']:
                self.profile_data[key] = value
        
        # Recalculate derived metrics
        if 'height_cm' in updates or 'weight_kg' in updates:
            self.profile_data['bmi'] = self._calculate_bmi(
                self.profile_data['height_cm'], 
                self.profile_data['weight_kg']
            )
            self.profile_data['bmi_category'] = self._get_bmi_category(self.profile_data['bmi'])
        
        if 'age' in updates:
            self.profile_data['age_group'] = self._get_age_group(self.profile_data['age'])
        
        # Update metadata
        self.profile_data['last_updated'] = datetime.now().isoformat()
        self.profile_data['profile_completeness'] = self._calculate_profile_completeness(self.profile_data)
        
        return {
            'success': True,
            'profile': self.profile_data,
            'insights': self._generate_profile_insights(self.profile_data)
        }
    
    def get_profile_summary(self):
        """Get a summary of the user profile"""
        if not self.profile_data:
            return {'exists': False, 'message': 'No profile created yet'}
        
        profile = self.profile_data
        
        summary = {
            'exists': True,
            'basic_info': {
                'age': profile['age'],
                'gender': profile['gender'],
                'age_group': profile['age_group']['description']
            },
            'physical_metrics': {
                'height_cm': profile['height_cm'],
                'weight_kg': profile['weight_kg'],
                'bmi': profile['bmi'],
                'bmi_category': profile['bmi_category']['description']
            },
            'health_status': {
                'known_conditions': len(profile.get('known_conditions', [])),
                'medications': len(profile.get('medications', [])),
                'allergies': len(profile.get('allergies', []))
            },
            'profile_completeness': profile['profile_completeness'],
            'last_updated': profile['last_updated']
        }
        
        return summary


# ================================
# MOCK DATA GENERATORS FOR INTEGRATED APPS
# ================================
class MockAppDataGenerator:
    def __init__(self):
        self.mindmap_data = None
        self.skintrack_data = None
        self.gastroguard_data = None
    
    def generate_mindmap_data(self, num_days=60):
        """Generate mock MindMap data"""
        np.random.seed(42)
        base_date = datetime(2025, 8, 1)
        
        data = []
        for i in range(num_days):
            date = base_date + timedelta(days=i)
            
            # Generate correlated mental health data
            mood_score = np.random.normal(6.5, 1.5)
            mood_score = max(1, min(10, mood_score))
            
            anxiety_score = np.random.normal(5.0, 2.0)
            anxiety_score = max(1, min(10, anxiety_score))
            
            depression_score = np.random.normal(4.5, 1.8)
            depression_score = max(1, min(10, depression_score))
            
            meditation_minutes = np.random.normal(15, 10)
            meditation_minutes = max(0, meditation_minutes)
            
            cognitive_fatigue = np.random.normal(5.5, 1.5)
            cognitive_fatigue = max(1, min(10, cognitive_fatigue))
            
            medication_effectiveness = np.random.normal(7.0, 1.2)
            medication_effectiveness = max(1, min(10, medication_effectiveness))
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'mood_score': round(mood_score, 1),
                'anxiety_score': round(anxiety_score, 1),
                'depression_score': round(depression_score, 1),
                'meditation_minutes': round(meditation_minutes, 1),
                'cognitive_fatigue': round(cognitive_fatigue, 1),
                'medication_effectiveness': round(medication_effectiveness, 1)
            })
        
        self.mindmap_data = pd.DataFrame(data)
        return self.mindmap_data
    
    def generate_skintrack_data(self, num_days=60):
        """Generate mock SkinTrack+ data"""
        np.random.seed(42)
        base_date = datetime(2025, 8, 1)
        
        data = []
        for i in range(num_days):
            date = base_date + timedelta(days=i)
            
            # Generate skin health data
            flare_severity = np.random.normal(4.0, 2.0)
            flare_severity = max(0, min(10, flare_severity))
            
            itch_intensity = np.random.normal(3.5, 1.8)
            itch_intensity = max(0, min(10, itch_intensity))
            
            treatment_effectiveness = np.random.normal(6.5, 1.5)
            treatment_effectiveness = max(1, min(10, treatment_effectiveness))
            
            skin_hydration = np.random.normal(6.0, 1.2)
            skin_hydration = max(1, min(10, skin_hydration))
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'flare_severity': round(flare_severity, 1),
                'itch_intensity': round(itch_intensity, 1),
                'treatment_effectiveness': round(treatment_effectiveness, 1),
                'skin_hydration': round(skin_hydration, 1)
            })
        
        self.skintrack_data = pd.DataFrame(data)
        return self.skintrack_data
    
    def generate_gastroguard_data(self, num_days=60):
        """Generate mock GastroGuard data"""
        np.random.seed(42)
        base_date = datetime(2025, 8, 1)
        
        data = []
        for i in range(num_days):
            date = base_date + timedelta(days=i)
            
            # Generate digestive health data
            gi_symptom_severity = np.random.normal(3.5, 1.8)
            gi_symptom_severity = max(0, min(10, gi_symptom_severity))
            
            food_sensitivity_score = np.random.normal(4.0, 1.5)
            food_sensitivity_score = max(1, min(10, food_sensitivity_score))
            
            digestive_comfort = np.random.normal(6.5, 1.2)
            digestive_comfort = max(1, min(10, digestive_comfort))
            
            medication_adherence = np.random.normal(8.0, 1.0)
            medication_adherence = max(1, min(10, medication_adherence))
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'gi_symptom_severity': round(gi_symptom_severity, 1),
                'food_sensitivity_score': round(food_sensitivity_score, 1),
                'digestive_comfort': round(digestive_comfort, 1),
                'medication_adherence': round(medication_adherence, 1)
            })
        
        self.gastroguard_data = pd.DataFrame(data)
        return self.gastroguard_data


# ================================
# SMART NOTIFICATIONS SERVICE
# ================================
class SmartNotificationsService:
    def __init__(self):
        self.notification_templates = {
            'morning': [
                "Good morning! How did you sleep last night?",
                "Rise and shine! Time to log your sleep quality.",
                "Morning! Ready to track your health today?"
            ],
            'evening': [
                "Time for your evening check-in! How are you feeling?",
                "Evening reflection time - how was your day?",
                "Don't forget to log your mood and stress levels!"
            ],
            'weekly': [
                "Weekly health check-in! Review your progress and set goals.",
                "Time for your weekly health review!",
                "How has your week been? Let's track your progress."
            ],
            'streak': [
                "🔥 Amazing! You've logged {days} days in a row! Keep it up!",
                "📈 {days} day streak! You're building a great habit!",
                "⭐ {days} days of consistent tracking! You're doing great!"
            ],
            'insight': [
                "💡 New health insight: {insight}",
                "📊 Your data shows: {insight}",
                "🔍 Interesting pattern detected: {insight}"
            ]
        }
    
    def generate_reminder(self, reminder_type, **kwargs):
        """Generate personalized reminder message"""
        templates = self.notification_templates.get(reminder_type, [])
        if not templates:
            return "Time for your health check-in!"
        
        template = random.choice(templates)
        return template.format(**kwargs)
    
    def should_send_adaptive_reminder(self, last_log_date, current_time):
        """Determine if adaptive reminder should be sent"""
        if not last_log_date:
            return True
        
        last_log = pd.Timestamp(last_log_date)
        today = pd.Timestamp.now().date()
        
        # Check if user hasn't logged today
        if last_log.date() != today:
            current_hour = current_time.hour
            
            # Send adaptive reminder based on time of day
            if current_hour >= 9 and current_hour < 12:
                return True, 'morning'
            elif current_hour >= 18 and current_hour < 22:
                return True, 'evening'
        
        return False, None


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    print("=" * 80)
    print("ADVANCED SLEEP TRACKER & STRESS/MOOD LOGGER ML ANALYTICS")
    print("=" * 80)
    print()
    
    init_dataset()

    # Generate sample data
    generate_sample_data(60)
    
    # Load and process data
    df = build_features()
    if df.empty:
        print("No data available yet.")
        exit()

    print(f"Loaded {len(df)} days of data")
    print(f"Features: {len(df.columns)} total features")
    print()

    # Initialize all new services
    insights_service = InsightsService()
    coping_service = CopingStrategiesService()
    gamification_service = GamificationService()
    trend_service = TrendAnalysisService()
    voice_service = VoiceTranscriptionService()
    notification_service = SmartNotificationsService()
    social_service = SocialSharingService()
    customization_service = CustomizationService()
    longterm_service = LongTermValueService()
    crossapp_service = CrossAppIntegrationService()
    mock_data_generator = MockAppDataGenerator()
    profile_service = UserProfileService()

    # Define targets for multi-task learning
    target_cols = {
        "regression": "symptoms.gi_flare",
        "classification": "symptoms.gi_flare"
    }
    feature_cols = [c for c in df.columns if c not in ["symptoms.gi_flare", "mood.mood_score"]]
    
    print(f"Training features: {len(feature_cols)}")
    print(f"Target: {target_cols['regression']} (regression + classification)")
    print()

    # Initialize experiment tracking (optional)
    tracker = None
    if WANDB_AVAILABLE or MLFLOW_AVAILABLE:
        tracker = ExperimentTracker(Config)
        tracker.init_tracking("transformer_multitask")
    else:
        print("Experiment tracking disabled (install wandb/mlflow for tracking)")

    # Train multi-task model
    model_path = os.path.join(Config.MODEL_DIR, "multitask_transformer.pt")
    success = train_multitask_model(df, feature_cols, target_cols, model_path, tracker)

    if success:
        print(f"\nTrained multi-task transformer model successfully")
        
        # Predict next day
        gi_reg, gi_cls = predict_next_multitask(df, model_path)
        mood_reg, _ = predict_next_multitask(df, model_path.replace("multitask_transformer.pt", "mood_model.pt"))
        
        preds = {
            "gi_flare_reg": gi_reg,
            "gi_flare_cls": gi_cls,
            "mood_reg": mood_reg
        }
        
        print("\n🔮 Multi-task Predictions for tomorrow:")
        if gi_reg is not None:
            print(f"  GI Flare (regression): {gi_reg:.2f}/10")
        if gi_cls is not None:
            print(f"  GI Flare (classification): {gi_cls:.1%} probability")
        if mood_reg is not None:
            print(f"  Mood: {mood_reg:.2f}/10")
        print()

        # Generate enhanced alerts
        alerts = generate_enhanced_alerts(df, preds)
        if alerts:
            print("🚨 Enhanced Health Alerts:")
            for alert in alerts:
                print(f"  {alert}")
        else:
            print("✅ No health alerts - looking good!")
    
    # ================================
    # NEW FEATURES DEMONSTRATION
    # ================================
    print("\n" + "=" * 80)
    print("🎯 NEW FEATURES DEMONSTRATION")
    print("=" * 80)
    
    # 1. Personalized Insights
    print("\n📊 PERSONALIZED INSIGHTS:")
    insights = insights_service.generate_insights(df)
    if insights:
        for insight in insights[:3]:  # Show top 3 insights
            print(f"  • {insight['title']}: {insight['description']}")
            if insight.get('recommendations'):
                print(f"    Recommendations: {', '.join(insight['recommendations'][:2])}")
    else:
        print("  No insights available yet (need more data)")
    
    # 2. Coping Strategies
    print("\n🧘 COPING STRATEGIES:")
    current_mood = df['mood.mood_score'].iloc[-1] if 'mood.mood_score' in df.columns and len(df) > 0 else 5
    current_stress = df['mood.stress_score'].iloc[-1] if 'mood.stress_score' in df.columns and len(df) > 0 else 5
    current_energy = 5  # Default energy level
    
    strategies = coping_service.get_recommended_strategies(current_mood, current_stress, current_energy)
    for strategy in strategies[:2]:  # Show top 2 strategies
        print(f"  • {strategy['name']} ({strategy['duration']} min)")
        print(f"    {strategy['description']}")
        print(f"    Effectiveness: {strategy['effectiveness']}/5")
    
    # 3. Mood Boosters
    print("\n😊 MOOD BOOSTERS:")
    mood_boosters = coping_service.get_recommended_mood_boosters(current_mood, current_energy)
    for booster in mood_boosters[:2]:  # Show top 2 boosters
        print(f"  • {booster['name']} ({booster['duration']} min)")
        print(f"    {booster['description']}")
        print(f"    Mood Impact: {booster['mood_impact']}/5")
    
    # 4. Gamification
    print("\n🏆 GAMIFICATION & PROGRESS:")
    user_stats = gamification_service.update_stats(df)
    print(f"  Level: {user_stats['level']} ({user_stats['experience']}/{user_stats['next_level_exp']} XP)")
    print(f"  Current Streak: {user_stats['streaks']['current_streak']} days")
    print(f"  Longest Streak: {user_stats['streaks']['longest_streak']} days")
    print(f"  Total Logs: {user_stats['total_logs']}")
    print(f"  Average Mood: {user_stats['average_mood']:.1f}/10")
    print(f"  Average Sleep Quality: {user_stats['average_sleep']:.1f}/10")
    print(f"  Average Stress: {user_stats['average_stress']:.1f}/10")
    
    # Show unlocked badges
    unlocked_badges = [badge for badge in user_stats['badges'] if badge['is_unlocked']]
    if unlocked_badges:
        print(f"  🎉 Unlocked Badges: {', '.join([badge['name'] for badge in unlocked_badges])}")
    
    # 5. Trend Analysis
    print("\n📈 TREND ANALYSIS:")
    trend_data = trend_service.generate_trend_charts(df, 'week')
    if trend_data:
        print(f"  Weekly Sleep Duration: {np.mean(trend_data['sleep_duration']):.1f} hours")
        print(f"  Weekly Sleep Quality: {np.mean(trend_data['sleep_quality']):.1f}/10")
        print(f"  Weekly Mood: {np.mean(trend_data['mood_score']):.1f}/10")
        print(f"  Weekly Stress: {np.mean(trend_data['stress_score']):.1f}/10")
    
    # 6. Voice Transcription Simulation
    print("\n🎤 VOICE TRANSCRIPTION SIMULATION:")
    transcription = voice_service.simulate_transcription(30)
    sentiment = voice_service.analyze_sentiment(transcription)
    print(f"  Transcribed: \"{transcription}\"")
    print(f"  Sentiment: {sentiment}")
    
    # 7. Smart Notifications
    print("\n🔔 SMART NOTIFICATIONS:")
    morning_reminder = notification_service.generate_reminder('morning')
    evening_reminder = notification_service.generate_reminder('evening')
    streak_reminder = notification_service.generate_reminder('streak', days=user_stats['streaks']['current_streak'])
    
    print(f"  Morning: \"{morning_reminder}\"")
    print(f"  Evening: \"{evening_reminder}\"")
    print(f"  Streak: \"{streak_reminder}\"")
    
    # 8. Correlation Analysis
    print("\n🔗 CORRELATION ANALYSIS:")
    heatmap_data = trend_service.generate_correlation_heatmap(df)
    if heatmap_data:
        print("  Key Correlations Found:")
        correlations = heatmap_data['correlations']
        labels = heatmap_data['labels']
        
        # Find strongest correlations
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels[i+1:], i+1):
                corr_value = correlations[i][j]
                if abs(corr_value) > 0.5:  # Strong correlation
                    direction = "positive" if corr_value > 0 else "negative"
                    print(f"    {label1} ↔ {label2}: {direction} ({corr_value:.2f})")
    
    # ================================
    # NEW ADVANCED FEATURES DEMONSTRATION
    # ================================
    print("\n" + "=" * 80)
    print("🌟 ADVANCED FEATURES DEMONSTRATION")
    print("=" * 80)
    
    # 9. Social & Sharing Features
    print("\n🌍 SOCIAL & SHARING:")
    sleep_benchmark = social_service.generate_community_benchmark(df, 'sleep_quality')
    if sleep_benchmark:
        print(f"  {sleep_benchmark['message']}")
        print(f"  Your sleep quality: {sleep_benchmark['user_average']:.1f}/10")
        print(f"  Community average: {sleep_benchmark['community_average']:.1f}/10")
        print(f"  Percentile: {sleep_benchmark['percentile']:.0f}th")
    
    # Generate export report
    medical_report = social_service.generate_export_report(df, 'medical_report', '3_months')
    if medical_report:
        print(f"  📄 Generated {medical_report['title']} for medical appointments")
        print(f"  📊 Report includes: {', '.join(medical_report['sections'].keys())}")
    
    # 10. Customization Features
    print("\n🎨 CUSTOMIZATION:")
    available_widgets = customization_service.get_available_widgets('sleep')
    print(f"  Available sleep widgets: {len(available_widgets)}")
    for widget in available_widgets[:3]:
        print(f"    • {widget['name']} ({widget['size']})")
    
    # Create custom dashboard
    custom_dashboard = customization_service.create_custom_dashboard(['sleep_graph', 'mood_meter', 'flare_risk'])
    print(f"  🎛️ Custom dashboard created with {len(custom_dashboard['widgets'])} widgets")
    
    # Theme configuration
    dark_theme = customization_service.get_theme_config('dark')
    print(f"  🌙 Dark theme available: {dark_theme['name']}")
    print(f"    Primary color: {dark_theme['primary']}")
    
    # 11. Long-Term Value Features
    print("\n📈 LONG-TERM VALUE:")
    
    # Goal setting
    sleep_goal = longterm_service.create_goal('sleep_duration', 7.5)
    if sleep_goal:
        print(f"  🎯 Created goal: {sleep_goal['name']}")
        print(f"    Target: {sleep_goal['target_value']} {sleep_goal['unit']}")
        
        # Update goal progress
        updated_goal = longterm_service.update_goal_progress(sleep_goal, df)
        print(f"    Current progress: {updated_goal['progress_percentage']:.1f}%")
        print(f"    Status: {updated_goal['status']}")
    
    # Generate personalized reports
    weekly_report = longterm_service.generate_personalized_report(df, 'weekly_summary', '7_days')
    if weekly_report:
        print(f"  📋 Generated {weekly_report['name']}")
        if 'highlights' in weekly_report['sections']:
            highlights = weekly_report['sections']['highlights']
            print(f"    Highlights: {len(highlights)} key insights")
            for highlight in highlights[:2]:
                print(f"      • {highlight}")
    
    monthly_report = longterm_service.generate_personalized_report(df, 'monthly_report', '30_days')
    if monthly_report:
        print(f"  📊 Generated {monthly_report['name']}")
        if 'overview' in monthly_report['sections']:
            overview = monthly_report['sections']['overview']
            print(f"    Data completeness: {overview['data_completeness']:.1f}%")
            print(f"    Total days analyzed: {overview['total_days']}")
    
    # AI Summary
    print("\n🤖 AI SUMMARY:")
    if weekly_report and 'insights' in weekly_report['sections']:
        ai_insights = weekly_report['sections']['insights']
        for insight in ai_insights[:2]:
            print(f"  • {insight}")
    
    # 12. Privacy & Security Features
    print("\n🔒 PRIVACY & SECURITY:")
    print("  ✅ On-device encryption: All logs stay private")
    print("  ✅ Local-first option: Choose not to sync to cloud")
    print("  ✅ Granular sharing controls: Share only selected data")
    print("  ✅ Anonymous community benchmarks: No personal data shared")
    
    # 13. Mood Reflection Prompts
    print("\n💭 MOOD REFLECTION PROMPTS:")
    reflection_prompts = [
        "What was one thing that made you smile today?",
        "How did you feel when you woke up this morning?",
        "What's one thing you're grateful for right now?",
        "How would you describe your energy level today?",
        "What's one small win you had today?"
    ]
    selected_prompt = random.choice(reflection_prompts)
    print(f"  💡 Today's prompt: \"{selected_prompt}\"")
    
    # ================================
    # USER PROFILE DEMONSTRATION
    # ================================
    print("\n" + "=" * 80)
    print("👤 USER PROFILE DEMONSTRATION")
    print("=" * 80)
    
    # Create sample user profile
    print("\n📝 CREATING USER PROFILE:")
    sample_profile = {
        'age': 28,
        'gender': 'female',
        'height_cm': 165,
        'weight_kg': 62,
        'known_conditions': ['anxiety', 'ibs', 'migraine'],
        'medications': ['anxiety_medication', 'probiotics'],
        'allergies': ['lactose_intolerance'],
        'activity_level': 'moderate'
    }
    
    profile_result = profile_service.create_user_profile(sample_profile)
    
    if profile_result['success']:
        profile = profile_result['profile']
        print(f"  ✅ Profile created successfully!")
        print(f"    • Age: {profile['age']} ({profile['age_group']['description']})")
        print(f"    • Gender: {profile['gender']}")
        print(f"    • Height: {profile['height_cm']} cm")
        print(f"    • Weight: {profile['weight_kg']} kg")
        print(f"    • BMI: {profile['bmi']} ({profile['bmi_category']['description']})")
        print(f"    • Known Conditions: {len(profile['known_conditions'])}")
        print(f"    • Profile Completeness: {profile['profile_completeness']}%")
        
        # Display profile insights
        print("\n💡 PROFILE INSIGHTS:")
        for insight in profile_result['insights']:
            print(f"    • {insight['title']}: {insight['description']}")
            print(f"      Recommendation: {insight['recommendation']}")
            print(f"      Priority: {insight['priority']}")
        
        # Get personalized recommendations
        print("\n🎯 PERSONALIZED RECOMMENDATIONS:")
        recommendations = profile_service.get_personalized_recommendations(profile, df)
        for rec in recommendations:
            print(f"    • {rec['title']}: {rec['description']}")
            print(f"      Action: {rec['action']}")
        
        # Profile summary
        print("\n📊 PROFILE SUMMARY:")
        summary = profile_service.get_profile_summary()
        print(f"    • Basic Info: {summary['basic_info']['age']} year old {summary['basic_info']['gender']}")
        print(f"    • Physical: {summary['physical_metrics']['height_cm']}cm, {summary['physical_metrics']['weight_kg']}kg")
        print(f"    • Health Status: {summary['health_status']['known_conditions']} conditions, {summary['health_status']['medications']} medications")
        print(f"    • Completeness: {summary['profile_completeness']}%")
    else:
        print(f"  ❌ Profile creation failed: {profile_result['error']}")
    
    # ================================
    # CROSS-APP INTEGRATION DEMONSTRATION
    # ================================
    print("\n" + "=" * 80)
    print("🔗 CROSS-APP INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Generate mock data for integrated apps
    print("\n📱 GENERATING MOCK DATA FOR INTEGRATED APPS:")
    mindmap_data = mock_data_generator.generate_mindmap_data(60)
    skintrack_data = mock_data_generator.generate_skintrack_data(60)
    gastroguard_data = mock_data_generator.generate_gastroguard_data(60)
    
    print(f"  🧠 MindMap data: {len(mindmap_data)} days of mental health tracking")
    print(f"  🩹 SkinTrack+ data: {len(skintrack_data)} days of skin health tracking")
    print(f"  🫀 GastroGuard data: {len(gastroguard_data)} days of digestive health tracking")
    
    # 14. Cross-App Data Synchronization
    print("\n🔄 CROSS-APP DATA SYNCHRONIZATION:")
    mindmap_sync = crossapp_service.sync_app_data('mindmap', df)
    skintrack_sync = crossapp_service.sync_app_data('skintrack', df)
    gastroguard_sync = crossapp_service.sync_app_data('gastroguard', df)
    
    if mindmap_sync:
        print(f"  ✅ MindMap sync: {mindmap_sync['data_points_synced']} data points")
        print(f"    Mapped fields: {list(mindmap_sync['mapped_fields'].keys())}")
    
    if skintrack_sync:
        print(f"  ✅ SkinTrack+ sync: {skintrack_sync['data_points_synced']} data points")
        print(f"    Mapped fields: {list(skintrack_sync['mapped_fields'].keys())}")
    
    if gastroguard_sync:
        print(f"  ✅ GastroGuard sync: {gastroguard_sync['data_points_synced']} data points")
        print(f"    Mapped fields: {list(gastroguard_sync['mapped_fields'].keys())}")
    
    # 15. Unified Health Dashboard
    print("\n📊 UNIFIED HEALTH DASHBOARD:")
    dashboard_data = crossapp_service.generate_unified_dashboard_data(
        df, mindmap_data, skintrack_data, gastroguard_data
    )
    
    print(f"  Connected apps: {', '.join(dashboard_data['apps_connected'])}")
    print(f"  Key metrics tracked: {len(dashboard_data['metrics'])}")
    
    # Display key metrics
    for metric, value in dashboard_data['metrics'].items():
        print(f"    • {metric.replace('_', ' ').title()}: {value:.1f}")
    
    # 16. Cross-App Correlations & Insights
    print("\n🔍 CROSS-APP CORRELATIONS & INSIGHTS:")
    unified_insights = crossapp_service.generate_unified_insights(
        df, mindmap_data, skintrack_data, gastroguard_data
    )
    
    if unified_insights:
        print(f"  Generated {len(unified_insights)} cross-app insights:")
        for insight in unified_insights[:5]:  # Show top 5 insights
            print(f"    • {insight['title']}: {insight['description']}")
            print(f"      Recommendation: {insight['recommendation']}")
            print(f"      Priority: {insight['priority']}")
    
    # 17. Holistic Health Analysis
    print("\n🌐 HOLISTIC HEALTH ANALYSIS:")
    
    # Calculate health scores across all apps
    health_scores = {}
    if 'sleep.quality_score' in df.columns:
        health_scores['Sleep'] = df['sleep.quality_score'].mean()
    if 'mood_score' in mindmap_data.columns:
        health_scores['Mental Health'] = mindmap_data['mood_score'].mean()
    if 'flare_severity' in skintrack_data.columns:
        health_scores['Skin Health'] = 10 - skintrack_data['flare_severity'].mean()
    if 'gi_symptom_severity' in gastroguard_data.columns:
        health_scores['Digestive Health'] = 10 - gastroguard_data['gi_symptom_severity'].mean()
    
    print("  Health Scores Across Apps:")
    for area, score in health_scores.items():
        status = "🟢 Excellent" if score >= 8 else "🟡 Good" if score >= 6 else "🔴 Needs Attention"
        print(f"    • {area}: {score:.1f}/10 {status}")
    
    overall_health = sum(health_scores.values()) / len(health_scores)
    print(f"  📈 Overall Health Score: {overall_health:.1f}/10")
    
    # 18. Treatment Optimization Insights
    print("\n💊 TREATMENT OPTIMIZATION INSIGHTS:")
    
    # Medication effectiveness analysis
    if 'medication_effectiveness' in mindmap_data.columns:
        med_effectiveness = mindmap_data['medication_effectiveness'].mean()
        print(f"  Mental Health Medication Effectiveness: {med_effectiveness:.1f}/10")
        if med_effectiveness < 6:
            print("    ⚠️ Consider discussing medication adjustments with healthcare provider")
        else:
            print("    ✅ Medication appears to be working well")
    
    # Treatment correlation analysis
    if 'treatment_effectiveness' in skintrack_data.columns:
        skin_treatment = skintrack_data['treatment_effectiveness'].mean()
        print(f"  Skin Treatment Effectiveness: {skin_treatment:.1f}/10")
        if skin_treatment < 6:
            print("    ⚠️ Consider reviewing skin treatment plan")
        else:
            print("    ✅ Skin treatments are effective")
    
    # 19. Lifestyle Recommendations
    print("\n🏃 LIFESTYLE RECOMMENDATIONS:")
    
    # Sleep-meditation correlation
    if 'meditation_minutes' in mindmap_data.columns and 'sleep.quality_score' in df.columns:
        meditation_avg = mindmap_data['meditation_minutes'].mean()
        sleep_quality = df['sleep.quality_score'].mean()
        if meditation_avg > 10 and sleep_quality > 7:
            print("  🧘 Continue meditation practice - it's improving your sleep quality")
        elif meditation_avg < 5:
            print("  🧘 Consider adding meditation to improve sleep and mental health")
    
    # Stress management across apps
    stress_sources = []
    if 'mood.stress_score' in df.columns and df['mood.stress_score'].mean() > 7:
        stress_sources.append("Sleep/Stress Tracker")
    if 'anxiety_score' in mindmap_data.columns and mindmap_data['anxiety_score'].mean() > 7:
        stress_sources.append("MindMap")
    
    if stress_sources:
        print(f"  ⚠️ High stress detected in: {', '.join(stress_sources)}")
        print("    💡 Consider stress management techniques across all health areas")
    else:
        print("  ✅ Stress levels are well-managed across all apps")
    
    # Finish experiment tracking
    if tracker:
        tracker.finish()
    
    print("\n" + "=" * 80)
    print("🎉 COMPLETE ADVANCED ANALYSIS WITH CROSS-APP INTEGRATION FINISHED!")
    print("=" * 80)
    print("\n✨ All features successfully integrated:")
    print("  🧠 Core ML Features:")
    print("    • Multi-task transformer with regression + classification")
    print("    • SHAP explainability for predictions")
    print("    • Data quality checks and outlier handling")
    print("    • Experiment tracking with W&B/MLflow")
    print("  📊 Analytics & Insights:")
    print("    • Personalized insights and recommendations")
    print("    • Trend analysis and correlation heatmaps")
    print("    • Long-term value reports and goal tracking")
    print("  🎮 User Experience:")
    print("    • Coping strategies and mood boosters")
    print("    • Gamification with badges and achievements")
    print("    • Voice transcription simulation")
    print("    • Smart notifications and reminders")
    print("  🌍 Social & Sharing:")
    print("    • Anonymous community benchmarks")
    print("    • Custom export reports for medical/therapy")
    print("    • Granular sharing controls")
    print("  🎨 Customization:")
    print("    • Customizable dashboards with widgets")
    print("    • Light/dark/blue-light themes")
    print("    • Mood reflection prompts")
    print("  🔒 Privacy & Security:")
    print("    • On-device encryption")
    print("    • Local-first data storage")
    print("    • Anonymous community comparisons")
    print("  🔗 Cross-App Integration:")
    print("    • MindMap mental health integration")
    print("    • SkinTrack+ skin health integration")
    print("    • GastroGuard digestive health integration")
    print("    • Unified health dashboard")
    print("    • Cross-app correlation analysis")
    print("    • Holistic health insights")
    print("    • Treatment optimization recommendations")
    print("  👤 User Profile System:")
    print("    • Comprehensive health profile creation")
    print("    • BMI calculation and health categorization")
    print("    • Age-group specific recommendations")
    print("    • Condition-specific insights and guidance")
    print("    • Personalized health recommendations")
    print("    • Profile completeness tracking")
    print("    • Gender-inclusive health considerations")
    print("\n🚀 Your comprehensive health ecosystem is ready!")
    print("   Sleep Tracker + MindMap + SkinTrack+ + GastroGuard + User Profiles = Complete Health Platform! 🎯")
