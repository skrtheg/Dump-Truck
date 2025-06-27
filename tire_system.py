import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import os

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Suppress TensorFlow warnings (these are from the original code)
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    TIRE_POSITIONS: List[str] = None
    DB_FILE: str = 'tire_system.db'
    SIM_INTERVAL: float = 1.0 # Reduced for faster updates
    TRAINING_INTERVAL_SECONDS: int = 20 # Train models every 20 seconds
    MIN_TRAINING_SAMPLES: int = 50 # Minimum samples needed to attempt ML training
    MAX_MEMORY_SAMPLES: int = 1000 # Limit data fetched from DB for ML training and display
    BATCH_SIZE: int = 16 # Added batch size for LSTM training

    def __post_init__(self):
        if self.TIRE_POSITIONS is None:
            self.TIRE_POSITIONS = ['LF', 'RF', 'LR', 'RR']

config = Config()

# Data Models
@dataclass
class TireReading:
    timestamp: datetime
    corner_id: str
    tire_pressure: float
    tread_depth: float # Added tread_depth
    tire_temp: float
    wheel_speed: float
    status: str # "Normal", "Warning", "Fault"
    is_fault_active_pressure: bool
    is_fault_active_overheating: bool # Added for overheating fault
    pressure_fault_severity_perc: float
    overheating_fault_severity_perc: float # Added for overheating fault
    fault_mode_pressure: str
    fault_mode_overheating: str # Added for overheating fault


@dataclass
class PredictionResult:
    classification: str # "Normal", "Warning", "Fault"
    rul_hours: float
    confidence: float
    timestamp: datetime


# Database Manager
class DatabaseManager:
    """Manages SQLite database for storing tire sensor readings."""
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Provides a context-managed database connection."""
        conn = sqlite3.connect(self.db_file, timeout=30.0)
        try:
            yield conn
        finally:
            conn.close()

    def init_database(self):
        """Initializes the database schema if tables do not exist."""
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tire_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    corner_id TEXT,
                    tire_pressure REAL,
                    tread_depth REAL, -- Added tread_depth
                    tire_temp REAL,
                    wheel_speed REAL,
                    status TEXT,
                    is_fault_active_pressure INTEGER,
                    is_fault_active_overheating INTEGER, -- Added for overheating
                    pressure_fault_severity_perc REAL,
                    overheating_fault_severity_perc REAL, -- Added for overheating
                    fault_mode_pressure TEXT,
                    fault_mode_overheating TEXT -- Added for overheating
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON tire_readings(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_corner ON tire_readings(corner_id)')
            conn.commit()

    def insert_readings(self, readings: List[TireReading]):
        """Inserts a list of TireReading objects into the database."""
        with self.get_connection() as conn:
            data = [(
                r.timestamp.isoformat(), r.corner_id, r.tire_pressure, r.tread_depth, # Added tread_depth
                r.tire_temp, r.wheel_speed, r.status, int(r.is_fault_active_pressure),
                int(r.is_fault_active_overheating), # Added for overheating
                r.pressure_fault_severity_perc, r.overheating_fault_severity_perc, # Added for overheating
                r.fault_mode_pressure, r.fault_mode_overheating # Added for overheating
            ) for r in readings]

            conn.executemany('''
                INSERT INTO tire_readings
                (timestamp, corner_id, tire_pressure, tread_depth, tire_temp, wheel_speed, -- Added tread_depth
                 status, is_fault_active_pressure, is_fault_active_overheating, -- Added for overheating
                 pressure_fault_severity_perc, overheating_fault_severity_perc, -- Added for overheating
                 fault_mode_pressure, fault_mode_overheating) -- Added for overheating
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) -- Updated number of placeholders
            ''', data)
            conn.commit()

    def get_recent_data(self, limit: int = 1000) -> pd.DataFrame:
        """Fetches the most recent tire readings from the database."""
        with self.get_connection() as conn:
            query = '''
                SELECT * FROM tire_readings
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(limit,))
            # Convert timestamp back to datetime and sort ascending for charts
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            return df

    def cleanup_old_data(self, days_to_keep: int = 7):
        """Deletes old data from the database to prevent it from growing too large."""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        with self.get_connection() as conn:
            conn.execute('DELETE FROM tire_readings WHERE timestamp < ?', (cutoff_date,))
            conn.commit()
            logger.info(f"Cleaned up data older than {days_to_keep} days.")

# Tire Sensor Simulator with Faults
class TireSensorSimulator:
    """
    Simulates real-time tire sensor data (pressure, tread depth, temperature)
    with capabilities for fault injection (sudden or gradual).
    """
    def __init__(self):
        self.time_step = 0
        self.base_tire_pressure = 35.0 # PSI (normal operating range for cars/light trucks)
        self.base_tread_depth = 8.0 # mm (new tire often 8-10mm)
        self.base_tire_temp = 25.0 # °C (ambient)

        self.fault_state = {
            'pressure_loss': {
                'active': False,
                'progress': 0.0, # Represents percentage drop from base pressure
                'start_time': None,
                'mode': 'gradual'
            },
            'tread_wear': {
                'active': False,
                'progress': 0.0, # Represents percentage wear from base depth (0% worn initially)
                'start_time': None,
                'mode': 'gradual'
            },
            'overheating': { # New fault type
                'active': False,
                'progress': 0.0, # Represents temperature increase above normal
                'start_time': None,
                'mode': 'gradual'
            }
        }

    def inject_fault(self, fault_type: str, mode: str = 'gradual'):
        """Inject a specific tire fault into the simulation."""
        if fault_type not in self.fault_state:
            logger.error(f"Error: Unknown fault type '{fault_type}'. Supported types: {list(self.fault_state.keys())}")
            return

        current_fault = self.fault_state[fault_type]
        current_fault['active'] = True
        current_fault['start_time'] = self.time_step
        current_fault['mode'] = mode

        # Reset other faults for a clearer demo (one fault at a time)
        for other_fault_type in self.fault_state:
            if other_fault_type != fault_type:
                self.fault_state[other_fault_type]['active'] = False
                self.fault_state[other_fault_type]['progress'] = 0.0 # Reset progress too
                self.fault_state[other_fault_type]['start_time'] = None

        if fault_type == 'pressure_loss':
            if mode == 'sudden':
                current_fault['progress'] = 60.0 # Sudden 60% pressure drop (relative)
            else: # gradual
                current_fault['progress'] = 5.0 # Start with 5% drop

        elif fault_type == 'tread_wear':
            if mode == 'sudden':
                current_fault['progress'] = 90.0 # 90% worn instantly
            else: # gradual
                current_fault['progress'] = 50.0 # Start at 50% worn

        elif fault_type == 'overheating':
            if mode == 'sudden':
                current_fault['progress'] = 80.0 # Sudden 80% temperature increase (relative)
            else: # gradual
                current_fault['progress'] = 20.0 # Start with 20% increase

    def reset_faults(self):
        """Resets all active tire faults and their internal states to normal."""
        for fault_type, fault_info in self.fault_state.items():
            fault_info['active'] = False
            fault_info['progress'] = 0.0
            fault_info['start_time'] = None
            fault_info['mode'] = 'gradual' # Reset to default mode

        self.time_step = 0

    def _update_fault_progression(self):
        """
        Updates the 'progress' of active gradual faults over each time step.
        'Sudden' faults maintain their set value without progression.
        """
        for fault_type, fault_info in self.fault_state.items():
            if fault_info['active'] and fault_info['mode'] == 'gradual':
                if fault_type == 'pressure_loss':
                    # Progress from 0 to 100 representing severity of pressure drop
                    fault_info['progress'] = min(100.0, fault_info['progress'] + np.random.uniform(0.1, 0.5))
                elif fault_type == 'tread_wear':
                    # Progress from 0 to 100 representing severity of tread wear
                    fault_info['progress'] = min(100.0, fault_info['progress'] + np.random.uniform(0.05, 0.2))
                elif fault_type == 'overheating':
                    # Progress from 0 to 100 representing severity of temperature increase
                    fault_info['progress'] = min(100.0, fault_info['progress'] + np.random.uniform(0.2, 0.8))

    def _determine_status(self, pressure: float, tread_depth: float, temp: float) -> str:
        """
        Determines the overall tire condition based on current sensor values and predefined thresholds.
        """
        # Define thresholds for pressure, tread depth (lower means more worn), and temperature
        # Normal ranges
        NORMAL_PRESSURE_MIN = 32.0
        NORMAL_PRESSURE_MAX = 38.0
        NORMAL_TREAD_DEPTH_MIN = 5.0 # Good tread
        NORMAL_TEMP_MAX = 40.0

        # Warning thresholds
        WARNING_PRESSURE_MIN = 28.0
        WARNING_TREAD_DEPTH_MIN = 3.0 # Moderate wear
        WARNING_TEMP_MAX = 55.0

        # Fault/Critical thresholds
        FAULT_PRESSURE_MIN = 25.0
        FAULT_TREAD_DEPTH_MIN = 1.6 # Legal limit or severely worn
        FAULT_TEMP_MAX = 70.0

        if pressure < FAULT_PRESSURE_MIN or \
           tread_depth < FAULT_TREAD_DEPTH_MIN or \
           temp > FAULT_TEMP_MAX:
            return "Fault"
        elif pressure < WARNING_PRESSURE_MIN or \
             tread_depth < WARNING_TREAD_DEPTH_MIN or \
             temp > WARNING_TEMP_MAX:
            return "Warning"
        else:
            return "Normal"

    def generate_tire_data(self) -> List[TireReading]:
        """
        Generates a single timestep of realistic tire sensor data for all tires,
        applying effects from any active faults.
        """
        self.time_step += 1
        self._update_fault_progression()

        readings = []
        timestamp = datetime.now()

        # Overall ambient noise/variation for all tires
        ambient_pressure_noise = np.random.normal(0, 0.2)
        ambient_temp_noise = np.random.normal(0, 0.5)
        ambient_speed_noise = np.random.normal(0, 0.1)

        for position in config.TIRE_POSITIONS:
            # Base values with minimal noise for normal operation
            current_pressure = self.base_tire_pressure + np.random.normal(0, 0.1) + ambient_pressure_noise
            current_tread_depth = self.base_tread_depth + np.random.normal(0, 0.02)
            current_temp = self.base_tire_temp + np.random.normal(0, 0.2) + ambient_temp_noise
            current_speed = 60.0 + np.random.normal(0, 0.5) + ambient_speed_noise # Simulating driving speed

            # Apply fault effects based on active faults
            pressure_fault_info = self.fault_state['pressure_loss']
            if pressure_fault_info['active']:
                # Pressure drop is relative to base pressure
                pressure_drop_percentage = pressure_fault_info['progress'] / 100.0
                current_pressure = self.base_tire_pressure * (1 - pressure_drop_percentage) + np.random.normal(0, 0.5) # More noise during fault
                if pressure_fault_info['mode'] == 'sudden':
                    current_pressure = self.base_tire_pressure * (1 - 0.6) + np.random.normal(0, 1.0) # Approx 60% sudden drop

            tread_wear_info = self.fault_state['tread_wear']
            if tread_wear_info['active']:
                # Tread wear reduces depth. Progress represents percentage wear.
                wear_percentage = tread_wear_info['progress'] / 100.0
                # Depth reduction relative to new tire depth (e.g., 8mm -> 0mm when 100% worn)
                current_tread_depth = self.base_tread_depth * (1 - wear_percentage) + np.random.normal(0, 0.05)
                if tread_wear_info['mode'] == 'sudden':
                    current_tread_depth = self.base_tread_depth * (1 - 0.9) + np.random.normal(0, 0.1) # Approx 90% sudden wear

            overheating_info = self.fault_state['overheating']
            if overheating_info['active']:
                # Temperature increase is relative to base temperature
                temp_increase_percentage = overheating_info['progress'] / 100.0
                # Max temp increase for simulation, e.g., 50C above base
                current_temp = self.base_tire_temp + (50.0 * temp_increase_percentage) + np.random.normal(0, 1.0)
                if overheating_info['mode'] == 'sudden':
                    current_temp = self.base_tire_temp + 40.0 + np.random.normal(0, 2.0) # Sudden 40C increase


            # Additional factors: temperature increases with speed and lower pressure
            current_temp += (current_speed / 100) * 5 # Higher speed, slightly higher temp
            if current_pressure < self.base_tire_pressure * 0.9: # Low pressure increases temp
                current_temp += (self.base_tire_pressure * 0.9 - current_pressure) * 0.5

            # Ensure realistic bounds and add minimal noise for all conditions
            current_pressure = max(10.0, min(50.0, current_pressure + np.random.normal(0, 0.05)))
            current_tread_depth = max(0.1, min(self.base_tread_depth, current_tread_depth + np.random.normal(0, 0.01)))
            current_temp = max(10.0, min(100.0, current_temp + np.random.normal(0, 0.1)))
            current_speed = max(0.0, min(120.0, current_speed + np.random.normal(0, 0.05))) # Speed can fluctuate

            status = self._determine_status(current_pressure, current_tread_depth, current_temp)

            reading = TireReading(
                timestamp=timestamp,
                corner_id=position,
                tire_pressure=round(current_pressure, 2),
                tread_depth=round(current_tread_depth, 2), # Use tread_depth
                tire_temp=round(current_temp, 2),
                wheel_speed=round(current_speed, 2),
                status=status,
                is_fault_active_pressure=pressure_fault_info['active'],
                is_fault_active_overheating=overheating_info['active'], # Check active status for overheating
                pressure_fault_severity_perc=round(pressure_fault_info['progress'], 1),
                overheating_fault_severity_perc=round(overheating_info['progress'], 1), # Store overheating progress
                fault_mode_pressure=pressure_fault_info['mode'] if pressure_fault_info['active'] else None,
                fault_mode_overheating=overheating_info['mode'] if overheating_info['active'] else None, # Store overheating mode
            )
            readings.append(reading)

        return readings

# Enhanced ML Pipeline
class TireMLPipeline:
    """Manages Machine Learning models for tire system status classification and RUL prediction."""
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.latest_prediction = None
        self.model_metrics = {}
        self.feature_columns = []

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepare features and labels for ML models.
        Returns scaled features (X), encoded labels (y), and original features DataFrame for reference.
        """
        base_features = ['tire_pressure', 'tread_depth', 'tire_temp', 'wheel_speed'] # Added tread_depth

        for col in base_features:
            if col not in df.columns:
                logger.warning(f"Missing column '{col}' in DataFrame for feature preparation. Filling with 0.")
                df[col] = 0.0

        df_processed = df.copy()

        if len(df_processed) > 0:
            df_processed['pressure_change_rate'] = df_processed.groupby('corner_id')['tire_pressure'].diff().fillna(0)
            df_processed['temp_change_rate'] = df_processed.groupby('corner_id')['tire_temp'].diff().fillna(0)
            df_processed['pressure_temp_ratio'] = df_processed['tire_pressure'] / (df_processed['tire_temp'] + 273.15)
            df_processed['speed_pressure_product'] = df_processed['wheel_speed'] * df_processed['tire_pressure']
            # New feature: tread wear rate (negative means getting worn)
            df_processed['tread_wear_rate'] = df_processed.groupby('corner_id')['tread_depth'].diff().fillna(0)
        else:
            df_processed['pressure_change_rate'] = 0.0
            df_processed['temp_change_rate'] = 0.0
            df_processed['pressure_temp_ratio'] = 0.0
            df_processed['speed_pressure_product'] = 0.0
            df_processed['tread_wear_rate'] = 0.0 # Initialize new feature

        self.feature_columns = [
            'tire_pressure', 'tread_depth', 'tire_temp', 'wheel_speed', # Added tread_depth
            'pressure_change_rate', 'temp_change_rate', 'tread_wear_rate', # Added tread_wear_rate
            'pressure_temp_ratio', 'speed_pressure_product'
        ]

        for col in self.feature_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0.0
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        X = self.scaler.fit_transform(df_processed[self.feature_columns])
        y = self.label_encoder.fit_transform(df_processed['status'])

        return X, y, df_processed

    def train_classification_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                     X_test: np.ndarray, y_test: np.ndarray):
        """Trains multiple classification models (SVM, RandomForest, DecisionTree)."""
        models_config = {
            'svm': SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),
            'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=10)
        }

        if len(np.unique(y_train)) < 2:
            logger.warning("Not enough unique classes in training data for classification models. Skipping training.")
            return

        for name, model in models_config.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                self.models[name] = model
                self.model_metrics[name] = {
                    'accuracy': accuracy,
                    'last_trained': datetime.now()
                }
                logger.info(f"{name} model trained with accuracy: {accuracy:.3f}")
            except Exception as e:
                logger.error(f"Error training {name} model: {e}")

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Builds a simple LSTM model for RUL prediction."""
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1, activation='linear')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def train_rul_model(self, X_all: np.ndarray, y_all: np.ndarray, df_original: pd.DataFrame):
        """
        Train LSTM model for Remaining Useful Life prediction.
        """
        try:
            # Generate synthetic RUL targets based on 'status' from df_original
            rul_targets = np.array([
                1000 + np.random.normal(0, 50) if status == 'Normal' else
                400 + np.random.normal(0, 30) if status == 'Warning' else
                100 + np.random.normal(0, 15)
                for status in df_original['status']
            ])

            sequence_length = 10
            X_seq, y_seq = [], []

            for i in range(sequence_length, len(X_all)):
                X_seq.append(X_all[i-sequence_length:i, :])
                y_seq.append(rul_targets[i])

            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)

            if X_seq.shape[0] == 0:
                logger.warning(f"Not enough data ({len(X_all)} samples) to create LSTM sequences (need >={sequence_length}). Skipping RUL training.")
                self.models['lstm'] = None
                return

            lstm_model = self.build_lstm_model((sequence_length, X_seq.shape[2]))

            early_stopping = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')

            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42
            )

            lstm_model.fit(
                X_train_lstm, y_train_lstm,
                validation_data=(X_test_lstm, y_test_lstm),
                epochs=50,
                batch_size=config.BATCH_SIZE,
                callbacks=[early_stopping],
                verbose=0
            )

            self.models['lstm'] = lstm_model
            logger.info("LSTM RUL model trained successfully")

        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            self.models['lstm'] = None

    def predict(self, df_current_window: pd.DataFrame) -> PredictionResult:
        """
        Make predictions (classification, RUL) using the trained models on the latest data.
        """
        if not self.feature_columns or df_current_window.empty:
            logger.warning("Feature columns not set or current data window is empty. Cannot predict.")
            return PredictionResult('Unknown', 0.0, 0.0, datetime.now())

        df_processed_window = df_current_window.copy()

        base_features = ['tire_pressure', 'tread_depth', 'tire_temp', 'wheel_speed'] # Added tread_depth
        for col in base_features:
            if col not in df_processed_window.columns:
                df_processed_window[col] = 0.0

        if len(df_processed_window) > 0:
            df_processed_window['pressure_change_rate'] = df_processed_window.groupby('corner_id')['tire_pressure'].diff().fillna(0)
            df_processed_window['temp_change_rate'] = df_processed_window.groupby('corner_id')['tire_temp'].diff().fillna(0)
            df_processed_window['pressure_temp_ratio'] = df_processed_window['tire_pressure'] / (df_processed_window['tire_temp'] + 273.15)
            df_processed_window['speed_pressure_product'] = df_processed_window['wheel_speed'] * df_processed_window['tire_pressure']
            df_processed_window['tread_wear_rate'] = df_processed_window.groupby('corner_id')['tread_depth'].diff().fillna(0) # New feature

        # Ensure all feature columns exist and are numeric
        for col in self.feature_columns:
            if col not in df_processed_window.columns:
                df_processed_window[col] = 0.0
            df_processed_window[col] = pd.to_numeric(df_processed_window[col], errors='coerce').fillna(0)

        latest_samples_per_tire = df_processed_window.groupby('corner_id').last()

        if latest_samples_per_tire.empty:
            return PredictionResult('Unknown', 0.0, 0.0, datetime.now())

        latest_features_combined = latest_samples_per_tire[self.feature_columns].mean().values.reshape(1, -1)

        try:
            X_scaled_latest = self.scaler.transform(latest_features_combined)
        except Exception as e:
            logger.error(f"Error scaling latest features for prediction: {e}")
            return PredictionResult('Error', 0.0, 0.0, datetime.now())

        # Classification prediction (ensemble voting)
        classifications = []
        confidences = []

        for name in ['svm', 'random_forest', 'decision_tree']:
            if name in self.models and self.models[name]:
                try:
                    pred = self.models[name].predict(X_scaled_latest)[0]
                    pred_proba = self.models[name].predict_proba(X_scaled_latest)[0]
                    classifications.append(pred)
                    confidences.append(max(pred_proba))
                except Exception as e:
                    logger.warning(f"Error getting prediction from {name}: {e}")

        status = 'Unknown'
        avg_confidence = 0.0
        if classifications:
            # Map numerical prediction back to label using label_encoder
            # Find the most common prediction (mode)
            from collections import Counter
            most_common_pred_num = Counter(classifications).most_common(1)[0][0]
            
            if hasattr(self.label_encoder, 'inverse_transform'):
                status = self.label_encoder.inverse_transform([most_common_pred_num])[0]
            else:
                logger.warning("LabelEncoder not fitted, using numerical class for status.")
                status = f"Class_{most_common_pred_num}"
            avg_confidence = np.mean(confidences) * 100 # Convert to percentage

        # RUL prediction (requires a sequence of data)
        rul_hours = 500.0 # Default RUL
        sequence_length = 10

        if 'lstm' in self.models and self.models['lstm'] is not None and len(df_processed_window) >= sequence_length * len(config.TIRE_POSITIONS):
            # To get sequential data, we need to ensure enough timesteps across all tires,
            # or just for one specific tire if we want tire-specific RUL.
            # For simplicity, let's average features over each timestamp.
            averaged_features_over_time = df_processed_window.groupby('timestamp')[self.feature_columns].mean().sort_index()

            if len(averaged_features_over_time) >= sequence_length:
                # Need to use the scaler that was fitted during training
                lstm_input_sequence = self.scaler.transform(averaged_features_over_time.tail(sequence_length)).reshape(1, sequence_length, len(self.feature_columns))
                try:
                    rul_pred = self.models['lstm'].predict(lstm_input_sequence, verbose=0)[0][0]
                    rul_hours = max(0, rul_pred)
                except Exception as e:
                    logger.warning(f"Error predicting RUL with LSTM: {e}")
            else:
                logger.info(f"Not enough sequential data ({len(averaged_features_over_time)} timesteps) for LSTM RUL prediction (need {sequence_length}).")

        self.latest_prediction = PredictionResult(
            classification=status,
            rul_hours=round(rul_hours, 1),
            confidence=round(avg_confidence, 1), # Round confidence to 1 decimal
            timestamp=datetime.now()
        )
        return self.latest_prediction

    def train_models(self):
        """Main training pipeline for all ML models."""
        try:
            df = self.db_manager.get_recent_data(limit=config.MAX_MEMORY_SAMPLES)

            if len(df) < config.MIN_TRAINING_SAMPLES:
                st.info(f"💡 Collecting data for ML training: {len(df)}/{config.MIN_TRAINING_SAMPLES} samples. Please wait.")
                self.models = {}
                self.latest_prediction = None
                return

            st.info(f"🏋️‍♀️ Training models with {len(df)} samples...")

            X, y, df_processed = self.prepare_features(df)

            if len(np.unique(y)) < 2:
                logger.warning("Only one unique class in target variable 'status'. Cannot perform stratified split. Using non-stratified split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

            self.train_classification_models(X_train, y_train, X_test, y_test)
            self.train_rul_model(X, y, df_processed)

            logger.info("Model training completed successfully.")
            if not df_processed.empty:
                self.predict(df_processed)

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            st.error(f"An error occurred during ML model training: {e}")


# --- Streamlit Application Logic ---

# Custom CSS for metric cards
st.markdown("""
<style>
    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .normal-card { background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%); }
    .warning-card { background: linear-gradient(90deg, #ff8008 0%, #ffc837 100%); }
    .fault-card { background: linear-gradient(90deg, #c33764 0%, #1d2671 100%); }
    .info-card { background: linear-gradient(90deg, #2193b0 0%, #6dd5ed 100%); }
</style>
""", unsafe_allow_html=True)

# Initialize simulator, DB manager, and ML pipeline in Streamlit's session state
# This ensures these objects persist across reruns without re-initializing unnecessarily
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager(config.DB_FILE)
    # Ensure cleanup is done once at startup or periodically
    st.session_state.db_manager.cleanup_old_data(days_to_keep=1) # Keep data for 1 day for demo purposes

if 'simulator' not in st.session_state:
    st.session_state.simulator = TireSensorSimulator()

if 'ml_pipeline' not in st.session_state:
    st.session_state.ml_pipeline = TireMLPipeline(st.session_state.db_manager)

if 'last_data_gen_time' not in st.session_state:
    st.session_state.last_data_gen_time = datetime.now()

if 'last_ml_train_time' not in st.session_state:
    st.session_state.last_ml_train_time = datetime.now()

# Alias for easier access
db_manager = st.session_state.db_manager
simulator = st.session_state.simulator
ml_pipeline = st.session_state.ml_pipeline

# Main function to run the Streamlit app for the tire system
def main():
    st.title("🛞 Advanced Tire System Monitoring & Predictive Maintenance")
    st.markdown("Real-time sensor data, fault injection, and AI-powered diagnostics for optimal tire health.")

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Tire System Control Panel")

        st.subheader("Data & Refresh")
        auto_refresh = st.checkbox("Auto Refresh Data", value=True, help="Automatically update sensor readings and dashboard.")
        refresh_rate = st.slider("Refresh Interval (seconds)", 0.5, 5.0, config.SIM_INTERVAL, 0.5, help="Time between dashboard updates when auto-refresh is on.")
        
        st.subheader("Fault Injection")
        fault_options = {
            'no_fault': '✅ No Faults', # Added no fault option
            'pressure_loss': '📉 Tire Pressure Loss',
            'tread_wear': '⚠️ Tire Tread Wear',
            'overheating': '🔥 Tire Overheating' # Added overheating fault
        }
        selected_fault = st.selectbox("Select Fault Type", list(fault_options.keys()), format_func=lambda x: fault_options[x])
        
        # Only show injection mode if a fault is selected
        fault_mode = "gradual"
        if selected_fault != 'no_fault':
            fault_mode = st.radio("Injection Mode", ["gradual", "sudden"], index=0)
        
        col_inject, col_reset = st.columns(2)
        with col_inject:
            if st.button(f"⚡ Apply Condition", type="secondary"): # Renamed button to be more general
                if selected_fault == 'no_fault':
                    simulator.reset_faults()
                    st.success("System set to No Faults (Normal Conditions)!")
                else:
                    simulator.inject_fault(selected_fault, fault_mode)
                    st.success(f"Applied {fault_options[selected_fault]} ({fault_mode})!")
        with col_reset:
            if st.button("🔄 Reset All System", type="primary"): # Renamed button for clarity
                simulator.reset_faults()
                st.session_state.db_manager.cleanup_old_data(days_to_keep=0) # Clear all data on full reset
                st.session_state.last_data_gen_time = datetime.now() # Reset data gen timer
                st.session_state.last_ml_train_time = datetime.now() # Reset ML train timer
                ml_pipeline.__init__(st.session_state.db_manager) # Re-initialize ML pipeline to clear models
                st.success("All system parameters and faults reset!")
                st.rerun() # Force a full dashboard refresh

        st.subheader("ML Model Training")
        if st.button("🏋️‍♀️ Train ML Models Now"):
            with st.spinner("Training ML models... This might take a moment if data is scarce."):
                ml_pipeline.train_models()
            if ml_pipeline.latest_prediction:
                st.success("ML models training complete and prediction updated!")
            else:
                st.warning("ML models trained, but no prediction could be made yet (e.g., insufficient data for prediction).")
        
        st.markdown("---")
        show_raw_data = st.checkbox("Show Raw Data Table", value=False)
        show_db_stats = st.checkbox("Show Database Stats", value=False) # Not directly used in display, can remove or use for debug


    # --- Data Generation & ML Training Logic (Streamlit's main loop) ---
    current_time = datetime.now()

    # Data generation
    if (current_time - st.session_state.last_data_gen_time).total_seconds() >= refresh_rate:
        new_readings = simulator.generate_tire_data()
        db_manager.insert_readings(new_readings)
        st.session_state.last_data_gen_time = current_time
        st.rerun() # Force a rerun to update the dashboard with new data

    # ML model training trigger
    if (current_time - st.session_state.last_ml_train_time).total_seconds() >= config.TRAINING_INTERVAL_SECONDS:
        ml_pipeline.train_models()
        st.session_state.last_ml_train_time = current_time
        # No explicit rerun needed here, as data generation already triggers it frequently


    # Fetch data for display
    df_all_data = db_manager.get_recent_data(limit=config.MAX_MEMORY_SAMPLES)

    if df_all_data.empty:
        st.info("⏳ Initializing tire monitoring system... Please wait for data to accumulate.")
        time.sleep(refresh_rate) # Small sleep to avoid busy-waiting too much
        st.rerun() # Re-run to check for data
        return

    # Prepare data for display and ML prediction
    # Ensure timestamp is datetime and sort for plotting
    df_all_data['timestamp'] = pd.to_datetime(df_all_data['timestamp'])
    df_all_data = df_all_data.sort_values('timestamp').reset_index(drop=True)
    
    latest_reading_overall = df_all_data.iloc[-1]
    
    # Get latest reading for each tire position for individual metrics
    latest_per_tire = df_all_data.groupby('corner_id').last()

    # --- Dashboard Tabs ---
    tab_overview, tab_pressure, tab_temp, tab_tread_depth, tab_speed, tab_predictive = st.tabs([ # Renamed tab
        "Overview", "Pressure", "Temperature", "Tread Depth", "Wheel Speed", "Predictive Analytics"
    ])

    with tab_overview:
        st.header("✨ Overall Tire System Health Overview")
        overall_health_status = latest_reading_overall['status']
        card_class = f"{overall_health_status.lower()}-card"
        st.markdown(f"""
        <div class="metric-card {card_class}">
            <h4>Overall System Status</h4>
            <h2>{overall_health_status}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Key Metrics Across All Tires")
        cols_metrics = st.columns(len(config.TIRE_POSITIONS))
        for i, pos in enumerate(config.TIRE_POSITIONS):
            if pos in latest_per_tire.index:
                tire_data = latest_per_tire.loc[pos]
                status_color = tire_data['status'].lower() # Not used for delta_color here
                cols_metrics[i].markdown(f"**{pos} Tire**")
                # Removed delta_color=status_color to resolve StreamlitAPIException
                cols_metrics[i].metric("Pressure (PSI)", f"{tire_data['tire_pressure']:.2f}")
                cols_metrics[i].metric("Tread Depth (mm)", f"{tire_data['tread_depth']:.2f}")
                cols_metrics[i].metric("Temperature (°C)", f"{tire_data['tire_temp']:.2f}")
                cols_metrics[i].metric("Wheel Speed (km/h)", f"{tire_data['wheel_speed']:.2f}")
            else:
                cols_metrics[i].info(f"No data for {pos} yet.")
        st.markdown("---")
        
        st.subheader("Recent Data Trends (All Tires)")
        # Create a single plot for all sensors across all tires, using facets for clarity
        fig_overview = make_subplots(rows=4, cols=1, 
                                     shared_xaxes=True, 
                                     vertical_spacing=0.1,
                                     subplot_titles=("Tire Pressure (PSI)", "Tread Depth (mm)", "Tire Temperature (°C)", "Wheel Speed (km/h)"))

        # Limit data for overview trends for better performance
        df_overview_trends = df_all_data.tail(200) 

        for i, position in enumerate(config.TIRE_POSITIONS):
            df_tire = df_overview_trends[df_overview_trends['corner_id'] == position]
            if not df_tire.empty:
                fig_overview.add_trace(go.Scatter(x=df_tire['timestamp'], y=df_tire['tire_pressure'], name=f'{position} Pressure', mode='lines', legendgroup='pressure', showlegend=True), row=1, col=1)
                fig_overview.add_trace(go.Scatter(x=df_tire['timestamp'], y=df_tire['tread_depth'], name=f'{position} Tread Depth', mode='lines', legendgroup='tread_depth', showlegend=True), row=2, col=1)
                fig_overview.add_trace(go.Scatter(x=df_tire['timestamp'], y=df_tire['tire_temp'], name=f'{position} Temp', mode='lines', legendgroup='temp', showlegend=True), row=3, col=1)
                fig_overview.add_trace(go.Scatter(x=df_tire['timestamp'], y=df_tire['wheel_speed'], name=f'{position} Speed', mode='lines', legendgroup='speed', hoverinfo='name+y+x'), row=4, col=1) # Corrected: Removed duplicated showlegend

        fig_overview.update_layout(height=800, title_text="Overview of Tire Sensor Trends Across All Positions")
        st.plotly_chart(fig_overview, use_container_width=True)


    with tab_pressure:
        st.header("💧 Tire Pressure Monitoring")
        st.markdown("Monitor real-time pressure for each tire, with visual indicators for normal, warning, and fault conditions.")
        for position in config.TIRE_POSITIONS:
            st.subheader(f"{position} Tire Pressure")
            df_tire_data = df_all_data[df_all_data['corner_id'] == position].tail(200) # Last 200 data points for current tire
            if not df_tire_data.empty:
                latest_pressure = df_tire_data.iloc[-1]['tire_pressure']
                status = df_tire_data.iloc[-1]['status']
                card_class = f"{status.lower()}-card"

                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <h4>Current Pressure ({position})</h4>
                    <h2>{latest_pressure:.2f} PSI</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Plotting individual pressure trend with thresholds
                fig_pressure = px.line(df_tire_data, x='timestamp', y='tire_pressure',
                                       title=f'{position} Tire Pressure Trend',
                                       labels={'tire_pressure': 'Pressure (PSI)', 'timestamp': 'Time'})
                # Add thresholds
                fig_pressure.add_hline(y=25.0, line_dash="dash", line_color="red", annotation_text="Fault (25 PSI)", annotation_position="bottom right")
                fig_pressure.add_hline(y=28.0, line_dash="dot", line_color="orange", annotation_text="Warning (28 PSI)", annotation_position="top left")
                fig_pressure.add_hline(y=32.0, line_dash="dot", line_color="green", annotation_text="Normal Min (32 PSI)", annotation_position="top right")
                fig_pressure.add_hline(y=38.0, line_dash="dot", line_color="green", annotation_text="Normal Max (38 PSI)", annotation_position="bottom left")

                fig_pressure.update_layout(height=400)
                st.plotly_chart(fig_pressure, use_container_width=True)
            else:
                st.info(f"No data for {position} tire pressure yet.")
            st.markdown("---") # Separator between tires


    with tab_temp:
        st.header("🌡️ Tire Temperature Monitoring")
        st.markdown("Track real-time temperature for each tire to detect overheating conditions.")
        for position in config.TIRE_POSITIONS:
            st.subheader(f"{position} Tire Temperature")
            df_tire_data = df_all_data[df_all_data['corner_id'] == position].tail(200)
            if not df_tire_data.empty:
                latest_temp = df_tire_data.iloc[-1]['tire_temp']
                status = df_tire_data.iloc[-1]['status']
                card_class = f"{status.lower()}-card"

                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <h4>Current Temperature ({position})</h4>
                    <h2>{latest_temp:.2f} °C</h2>
                </div>
                """, unsafe_allow_html=True)

                fig_temp = px.line(df_tire_data, x='timestamp', y='tire_temp',
                                   title=f'{position} Tire Temperature Trend',
                                   labels={'tire_temp': 'Temperature (°C)', 'timestamp': 'Time'})
                # Add thresholds
                fig_temp.add_hline(y=70.0, line_dash="dash", line_color="red", annotation_text="Fault (70°C)", annotation_position="top right")
                fig_temp.add_hline(y=55.0, line_dash="dot", line_color="orange", annotation_text="Warning (55°C)", annotation_position="top right")
                fig_temp.add_hline(y=40.0, line_dash="dot", line_color="green", annotation_text="Normal Max (40°C)", annotation_position="bottom left")

                fig_temp.update_layout(height=400)
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.info(f"No data for {position} tire temperature yet.")
            st.markdown("---")


    with tab_tread_depth: # Changed from tab_wear
        st.header("🛞 Tire Tread Depth Monitoring")
        st.markdown("Monitor the remaining tread depth to assess wear and plan for replacement.")
        for position in config.TIRE_POSITIONS:
            st.subheader(f"{position} Tire Tread Depth")
            df_tire_data = df_all_data[df_all_data['corner_id'] == position].tail(200)
            if not df_tire_data.empty:
                latest_tread_depth = df_tire_data.iloc[-1]['tread_depth']
                status = df_tire_data.iloc[-1]['status']
                card_class = f"{status.lower()}-card"

                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <h4>Current Tread Depth ({position})</h4>
                    <h2>{latest_tread_depth:.2f} mm</h2>
                </div>
                """, unsafe_allow_html=True)

                fig_tread = px.line(df_tire_data, x='timestamp', y='tread_depth',
                                    title=f'{position} Tire Tread Depth Trend',
                                    labels={'tread_depth': 'Tread Depth (mm)', 'timestamp': 'Time'})
                # Add thresholds (lower depth means more wear)
                fig_tread.add_hline(y=1.6, line_dash="dash", line_color="red", annotation_text="Fault (1.6 mm - Legal Limit)", annotation_position="bottom right")
                fig_tread.add_hline(y=3.0, line_dash="dot", line_color="orange", annotation_text="Warning (3.0 mm)", annotation_position="top right")
                fig_tread.add_hline(y=5.0, line_dash="dot", line_color="green", annotation_text="Normal Min (5.0 mm)", annotation_position="bottom left")
                fig_tread.add_hline(y=8.0, line_dash="dot", line_color="blue", annotation_text="New Tire Avg (8.0 mm)", annotation_position="top right")


                fig_tread.update_layout(height=400)
                st.plotly_chart(fig_tread, use_container_width=True)
            else:
                st.info(f"No data for {position} tire tread depth yet.")
            st.markdown("---")


    with tab_speed:
        st.header("⚡ Wheel Speed Monitoring")
        st.markdown("Monitor wheel speed for consistency and potential anomalies across tires.")
        for position in config.TIRE_POSITIONS:
            st.subheader(f"{position} Wheel Speed")
            df_tire_data = df_all_data[df_all_data['corner_id'] == position].tail(200)
            if not df_tire_data.empty:
                latest_speed = df_tire_data.iloc[-1]['wheel_speed']
                status = df_tire_data.iloc[-1]['status']
                card_class = f"{status.lower()}-card"

                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <h4>Current Wheel Speed ({position})</h4>
                    <h2>{latest_speed:.2f} km/h</h2>
                </div>
                """, unsafe_allow_html=True)

                fig_speed = px.line(df_tire_data, x='timestamp', y='wheel_speed',
                                    title=f'{position} Wheel Speed Trend',
                                    labels={'wheel_speed': 'Speed (km/h)', 'timestamp': 'Time'})
                fig_speed.update_layout(height=400)
                st.plotly_chart(fig_speed, use_container_width=True)
            else:
                st.info(f"No data for {position} wheel speed yet.")
            st.markdown("---")


    with tab_predictive:
        st.header("🧠 AI-Powered Predictive Analytics")
        st.markdown("Leveraging Machine Learning models to predict tire health, detect anomalies, and estimate Remaining Useful Life (RUL).")
        
        ml_pred = ml_pipeline.latest_prediction # Use the latest stored prediction
        
        if ml_pred and ml_pipeline.models:
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                status_color_map = {'Normal': 'green', 'Warning': 'orange', 'Fault': 'red', 'Unknown': 'gray', 'Error': 'red'}
                status_color_class = status_color_map.get(ml_pred.classification, 'gray')
                st.markdown(f"""
                <div class="metric-card {status_color_class}-card">
                    <h4>Overall Predicted Status</h4>
                    <h2>{ml_pred.classification}</h2>
                    <p>Confidence: {ml_pred.confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            with pred_col2:
                st.info(f"""
                **RUL Estimate**
                ## {ml_pred.rul_hours:,.0f} hours
                """)
            
            with pred_col3:
                st.info(f"""
                **Last Prediction Time**
                ## {ml_pred.timestamp.strftime('%H:%M:%S')}
                """)
            
            st.subheader("Model Performance Metrics")
            if ml_pipeline.model_metrics:
                metrics_cols = st.columns(len(ml_pipeline.model_metrics))
                for i, (model_name, metrics) in enumerate(ml_pipeline.model_metrics.items()):
                    metrics_cols[i].markdown(f"**{model_name.replace('_', ' ').title()}**")
                    metrics_cols[i].metric("Accuracy", f"{metrics['accuracy']:.2f}")
                    metrics_cols[i].caption(f"Last Trained: {metrics['last_trained'].strftime('%H:%M:%S')}")
            else:
                st.info("No ML model metrics available yet. Train models to see performance.")
            
            st.subheader("Predicted Status Trend")
            # Show the history of predicted status if available (simplistic for demo)
            # This would ideally come from storing ml_pipeline.latest_prediction over time
            # For this demo, let's just show the latest overall health status from simulator data
            status_counts = df_all_data['status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig_status = px.bar(status_counts, x='Status', y='Count', color='Status',
                                 title='Distribution of Recent Tire Health Status',
                                 color_discrete_map={'Normal': 'green', 'Warning': 'orange', 'Fault': 'red'})
            st.plotly_chart(fig_status, use_container_width=True)

        else:
            st.warning("ML models not yet trained or no predictions available. Please trigger training via the sidebar.")

    # Raw data table
    if show_raw_data:
        st.markdown("---")
        st.markdown("#### 📋 Raw Recent Sensor Data (Last 20 records)")
        st.dataframe(df_all_data.tail(20), use_container_width=True)

    # Database Stats (optional, for debug)
    if show_db_stats:
        st.markdown("---")
        st.markdown("#### 🗄️ Database Statistics")
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM tire_readings")
            count = cursor.fetchone()[0]
            st.write(f"Total records in database: {count}")
            cursor = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM tire_readings")
            min_ts, max_ts = cursor.fetchone()
            st.write(f"Data range: From {min_ts} to {max_ts}")


# Run the Streamlit application
if __name__ == "__main__":
    main()