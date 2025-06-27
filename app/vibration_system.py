import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import os

# Import for auto-refresh functionality
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# Suppress TensorFlow logging and other warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global constants
CSV_FILE = "engine_sensor_data.csv"
MIN_TRAINING_SAMPLES = 50

class EngineComponentSimulator:
    """
    Enhanced simulator with improved normal/fault behavior and sensor-wise monitoring
    """
    def __init__(self):
        self.time_step = 0
        # Refined engine states with more realistic noise levels
        self.engine_states = {
            'idle': {'rpm': (800, 900), 'load': 0.08, 'temp': (80, 85), 'noise_factor': 0.1},
            'normal': {'rpm': (1200, 1600), 'load': 0.35, 'temp': (85, 92), 'noise_factor': 0.15},
            'heavy': {'rpm': (1800, 2100), 'load': 0.75, 'temp': (90, 100), 'noise_factor': 0.25},
            'critical_op': {'rpm': (2000, 2300), 'load': 0.95, 'temp': (100, 110), 'noise_factor': 0.35}
        }
        
        # Enhanced fault management with multiple fault types
        self.fault_active = {
            'vibration_imbalance': False,
            'bearing_wear': False,
            'misalignment': False,
            'no_fault_mode': True  # Default to no fault mode
        }
        
        # Sensor health tracking
        self.sensor_status = {
            'accelerometer': {'status': 'Normal', 'last_anomaly': None},
            'gyroscope': {'status': 'Normal', 'last_anomaly': None},
            'temperature': {'status': 'Normal', 'last_anomaly': None},
            'rpm': {'status': 'Normal', 'last_anomaly': None}
        }

    def toggle_fault_mode(self, mode: str):
        """Toggle between fault and no-fault modes"""
        if mode == "No Faults":
            self.fault_active = {fault: False for fault in self.fault_active}
            self.fault_active['no_fault_mode'] = True
            st.success("✅ System set to Normal Operation Mode")
        elif mode == "Fault Conditions":
            self.fault_active['no_fault_mode'] = False
            st.info("⚠️ System set to Fault Condition Mode - Inject specific faults using controls")

    def inject_fault(self, fault_type: str):
        """Activates a specific fault in the simulation"""
        if fault_type in self.fault_active:
            self.fault_active[fault_type] = True
            self.fault_active['no_fault_mode'] = False
            fault_messages = {
                'vibration_imbalance': "🚨 Vibration Imbalance Fault Activated!",
                'bearing_wear': "🔧 Bearing Wear Fault Activated!",
                'misalignment': "⚖️ Shaft Misalignment Fault Activated!"
            }
            st.success(fault_messages.get(fault_type, f"Fault {fault_type} activated"))

    def reset_faults(self):
        """Resets all active faults to normal"""
        self.fault_active = {fault: False for fault in self.fault_active}
        self.fault_active['no_fault_mode'] = True
        self.sensor_status = {sensor: {'status': 'Normal', 'last_anomaly': None} 
                             for sensor in self.sensor_status}
        st.success("✅ All faults reset to normal operation.")

    def update_sensor_status(self, sensor_data):
        """Update individual sensor status based on current readings"""
        # Accelerometer status
        accel_rms = sensor_data['vibration_rms']
        if accel_rms > 3.0:
            self.sensor_status['accelerometer']['status'] = 'Fault'
            self.sensor_status['accelerometer']['last_anomaly'] = datetime.now()
        elif accel_rms > 1.5:
            self.sensor_status['accelerometer']['status'] = 'Warning'
        else:
            self.sensor_status['accelerometer']['status'] = 'Normal'
        
        # Gyroscope status
        gyro_magnitude = np.sqrt(sensor_data['gyro_x']**2 + sensor_data['gyro_y']**2 + sensor_data['gyro_z']**2)
        if gyro_magnitude > 8.0:
            self.sensor_status['gyroscope']['status'] = 'Fault'
            self.sensor_status['gyroscope']['last_anomaly'] = datetime.now()
        elif gyro_magnitude > 5.0:
            self.sensor_status['gyroscope']['status'] = 'Warning'
        else:
            self.sensor_status['gyroscope']['status'] = 'Normal'
        
        # Temperature status
        if sensor_data['engine_temp'] > 105:
            self.sensor_status['temperature']['status'] = 'Fault'
            self.sensor_status['temperature']['last_anomaly'] = datetime.now()
        elif sensor_data['engine_temp'] > 95:
            self.sensor_status['temperature']['status'] = 'Warning'
        else:
            self.sensor_status['temperature']['status'] = 'Normal'
        
        # RPM status
        if sensor_data['engine_rpm'] > 2300 or sensor_data['engine_rpm'] < 500:
            self.sensor_status['rpm']['status'] = 'Fault'
            self.sensor_status['rpm']['last_anomaly'] = datetime.now()
        elif sensor_data['engine_rpm'] > 2100 or sensor_data['engine_rpm'] < 700:
            self.sensor_status['rpm']['status'] = 'Warning'
        else:
            self.sensor_status['rpm']['status'] = 'Normal'

    def generate_sensor_data(self) -> dict:
        """
        Enhanced sensor data generation with improved normal/fault behavior
        """
        self.time_step += 1
        
        # Determine operating state
        states = ['idle', 'normal', 'heavy', 'critical_op']
        probs = [0.2, 0.5, 0.25, 0.05]
        current_operating_state = np.random.choice(states, p=probs)
        
        state_config = self.engine_states[current_operating_state]
        noise_factor = state_config['noise_factor']
        
        # Base engine parameters with reduced noise in no-fault mode
        rpm_range = state_config['rpm']
        if self.fault_active['no_fault_mode']:
            # Minimal noise in no-fault mode
            engine_rpm = np.random.uniform(*rpm_range) + np.random.normal(0, 10)
            engine_load = state_config['load'] + np.random.normal(0, 0.02)
            temp_range = state_config['temp']
            engine_temp = np.random.uniform(*temp_range) + np.random.normal(0, 1)
        else:
            # Normal noise levels
            engine_rpm = np.random.uniform(*rpm_range) + np.random.normal(0, 50)
            engine_load = state_config['load'] + np.random.normal(0, 0.05)
            temp_range = state_config['temp']
            engine_temp = np.random.uniform(*temp_range) + np.random.normal(0, 2)

        # Base sensor readings with operating state influence
        base_vib_factor = 0.3 + (engine_load * 0.8) if self.fault_active['no_fault_mode'] else 0.5 + (engine_load * 1.5)
        rpm_intensity_factor = (engine_rpm - 800) / 1600
        
        # Accelerometer data
        if self.fault_active['no_fault_mode']:
            # Minimal, realistic vibration in normal operation
            accel_x_base = np.random.normal(0, base_vib_factor * 0.3)
            accel_y_base = np.random.normal(0, base_vib_factor * 0.2)
            accel_z_base = np.random.normal(9.81, base_vib_factor * 0.1)
        else:
            accel_x_base = np.random.normal(0, base_vib_factor + rpm_intensity_factor * 0.3)
            accel_y_base = np.random.normal(0, base_vib_factor + rpm_intensity_factor * 0.2)
            accel_z_base = np.random.normal(9.81, base_vib_factor + rpm_intensity_factor * 0.4)

        # Gyroscope data
        if self.fault_active['no_fault_mode']:
            # Minimal angular velocity in normal operation
            gyro_x = np.random.normal(0, 0.5 + engine_load * 0.3)
            gyro_y = np.random.normal(0, 0.3 + engine_load * 0.2)
            gyro_z = np.random.normal(0, 0.2 + engine_load * 0.1)
        else:
            gyro_x = np.random.normal(0, 2 + engine_load)
            gyro_y = np.random.normal(0, 1.5 + engine_load)
            gyro_z = np.random.normal(0, 1 + engine_load)
        
        # Initialize final sensor values
        accel_x, accel_y, accel_z = accel_x_base, accel_y_base, accel_z_base
        vibration_skew = np.random.normal(0, 0.1) if self.fault_active['no_fault_mode'] else np.random.normal(0, 0.2)

        # Apply specific fault effects
        if self.fault_active['vibration_imbalance']:
            accel_x += np.random.normal(3.0, 0.5)
            accel_y += np.random.normal(3.0, 0.5)
            accel_z += np.random.normal(3.0, 0.5)
            vibration_skew = np.random.uniform(1.5, 2.5)
            
        if self.fault_active['bearing_wear']:
            # Bearing wear increases high-frequency noise
            accel_x += np.random.normal(0, 1.5)
            accel_y += np.random.normal(0, 1.5)
            accel_z += np.random.normal(0, 1.5)
            gyro_x += np.random.normal(0, 2.0)
            gyro_y += np.random.normal(0, 2.0)
            
        if self.fault_active['misalignment']:
            # Misalignment affects gyroscope readings more
            gyro_x += np.random.normal(0, 3.0)
            gyro_y += np.random.normal(0, 3.0)
            gyro_z += np.random.normal(0, 2.0)
            vibration_skew += np.random.uniform(0.5, 1.0)

        # Calculate derived metrics
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        vibration_rms = np.sqrt(np.mean([accel_x**2, accel_y**2, accel_z**2]))
        tilt_angle = np.degrees(np.arctan2(np.sqrt(accel_x**2 + accel_y**2), accel_z))
        vibration_rms = max(0.01, vibration_rms)

        # Enhanced anomaly score calculation
        if self.fault_active['no_fault_mode']:
            # Very low anomaly scores in normal operation
            anomaly_score = min(25, max(0, 
                (vibration_rms - 0.5) * 8 +
                abs(vibration_skew) * 5 +
                abs(tilt_angle - 2) * 1 +
                max(0, engine_temp - 90) * 0.5 +
                max(0, engine_load - 0.6) * 5
            ))
        else:
            anomaly_score = min(100, max(0, 
                (vibration_rms - 1.5) * 15 +
                (vibration_skew - 0.5) * 10 +
                abs(tilt_angle - 5) * 2 +
                max(0, engine_temp - 95) * 1.5 +
                max(0, engine_load - 0.8) * 10
            ))
        
        # Health status determination
        if anomaly_score > 60:
            health = 'Fault'
        elif anomaly_score > 30:
            health = 'Warning'
        else:
            health = 'Normal'
            
        # RUL estimation
        base_rul = 5000
        rul = max(0, base_rul - anomaly_score * 50)

        sensor_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'engine_rpm': round(engine_rpm, 0),
            'engine_load': round(engine_load, 2),
            'engine_temp': round(engine_temp, 2),
            'accel_x': round(accel_x, 3),
            'accel_y': round(accel_y, 3),
            'accel_z': round(accel_z, 3),
            'accel_magnitude': round(accel_mag, 3),
            'gyro_x': round(gyro_x, 3),
            'gyro_y': round(gyro_y, 3),
            'gyro_z': round(gyro_z, 3),
            'tilt_angle': round(tilt_angle, 2),
            'vibration_rms': round(vibration_rms, 3),
            'vibration_skew': round(vibration_skew, 3),
            'stability_index': round(max(0, 100 - anomaly_score), 2),
            'anomaly_score': round(anomaly_score, 2),
            'health_status': health,
            'rul_estimate': int(rul),
            'operating_state': current_operating_state
        }
        
        # Update sensor status
        self.update_sensor_status(sensor_data)
        
        return sensor_data

class MLPredictor:
    """
    Enhanced ML predictor with improved feature handling
    """
    def __init__(self):
        self.svm_model = None
        self.dt_model = None
        self.scaler = StandardScaler()
        self.models_trained = False
        self.health_labels = ['Normal', 'Warning', 'Fault']
        self.health_map = {label: i for i, label in enumerate(self.health_labels)}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and scale features for ML models"""
        feature_cols = [
            'engine_rpm', 'engine_load', 'engine_temp', 
            'accel_x', 'accel_y', 'accel_z', 
            'gyro_x', 'gyro_y', 'gyro_z', 
            'vibration_rms', 'vibration_skew', 'tilt_angle'
        ]
        
        X = df[feature_cols].copy()
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        return X

    def train_models(self, df: pd.DataFrame) -> bool:
        """Train ML models with enhanced error handling"""
        if len(df) < MIN_TRAINING_SAMPLES:
            st.info(f"Collecting more data for ML training: {len(df)}/{MIN_TRAINING_SAMPLES} samples.")
            return False
        
        try:
            X = self.prepare_features(df)
            y_health = df['health_status'].map(self.health_map).fillna(0)
            y_anomaly = (df['anomaly_score'] > 30).astype(int)
            
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models with better error handling
            if len(np.unique(y_health)) > 1:
                X_train_health, X_test_health, y_train_health, y_test_health = train_test_split(
                    X_scaled, y_health, test_size=0.2, random_state=42, stratify=y_health
                )
                self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
                self.svm_model.fit(X_train_health, y_train_health)
                svm_accuracy = accuracy_score(y_test_health, self.svm_model.predict(X_test_health))
                st.write(f"✅ SVM Health Classification Accuracy: {svm_accuracy:.2f}")
            else:
                st.warning("Not enough health status variations for SVM training.")
                self.svm_model = None
            
            if len(np.unique(y_anomaly)) > 1:
                X_train_anomaly, X_test_anomaly, y_train_anomaly, y_test_anomaly = train_test_split(
                    X_scaled, y_anomaly, test_size=0.2, random_state=42, stratify=y_anomaly
                )
                self.dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
                self.dt_model.fit(X_train_anomaly, y_train_anomaly)
                dt_accuracy = accuracy_score(y_test_anomaly, self.dt_model.predict(X_test_anomaly))
                st.write(f"✅ Decision Tree Anomaly Detection Accuracy: {dt_accuracy:.2f}")
            else:
                st.warning("Not enough anomaly variations for Decision Tree training.")
                self.dt_model = None

            self.models_trained = (self.svm_model is not None and self.dt_model is not None)
            return self.models_trained
            
        except Exception as e:
            st.error(f"Error during ML training: {e}")
            self.models_trained = False
            return False
    
    def predict(self, df_latest: pd.DataFrame) -> dict:
        """Make predictions on latest sensor data"""
        if not self.models_trained or df_latest.empty:
            return None
        
        try:
            latest_features = self.prepare_features(df_latest.tail(1))
            X_scaled_latest = self.scaler.transform(latest_features)
            
            health_pred_num = self.svm_model.predict(X_scaled_latest)[0]
            health_proba = self.svm_model.predict_proba(X_scaled_latest)[0]
            anomaly_pred_num = self.dt_model.predict(X_scaled_latest)[0]
            rul_pred = df_latest.iloc[-1]['rul_estimate']
            
            return {
                'health_prediction': self.health_labels[health_pred_num],
                'health_confidence': max(health_proba) * 100,
                'anomaly_detected': bool(anomaly_pred_num),
                'rul_prediction': int(rul_pred)
            }
        except Exception as e:
            st.error(f"Error during ML prediction: {e}")
            return None

# Initialize session state
if 'simulator' not in st.session_state:
    st.session_state.simulator = EngineComponentSimulator()
    st.session_state.predictor = MLPredictor()
    st.session_state.data = pd.DataFrame()
    st.session_state.last_data_gen_time = datetime.now()
    st.session_state.last_ml_train_time = datetime.now()

simulator = st.session_state.simulator
predictor = st.session_state.predictor

# Enhanced CSS for better visualization
st.markdown("""
<style>
    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .normal-card { background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); }
    .warning-card { background: linear-gradient(135deg, #ff8008 0%, #ffc837 100%); }
    .fault-card { background: linear-gradient(135deg, #c33764 0%, #1d2671 100%); }
    
    .sensor-status {
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        font-weight: bold;
        text-align: center;
    }
    .sensor-normal { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .sensor-warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
    .sensor-fault { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    
    .mode-indicator {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .no-fault-mode { background: linear-gradient(135deg, #4CAF50, #8BC34A); color: white; }
    .fault-mode { background: linear-gradient(135deg, #FF5722, #FF9800); color: white; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🚛 Enhanced Engine Vibration Monitoring System")
st.markdown("**Real-time sensor monitoring with fault/no-fault mode switching**")

# --- Enhanced Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ System Controls")
    
    # Operating Mode Selection
    st.markdown("### 🔧 Operating Mode")
    operation_mode = st.radio(
        "Select Operating Mode:",
        ["No Faults", "Fault Conditions"],
        help="Toggle between normal operation and fault condition modes"
    )
    
    if st.button("Apply Mode", key="apply_mode_btn"):
        simulator.toggle_fault_mode(operation_mode)
    
    # Current mode indicator
    current_mode = "No Faults" if simulator.fault_active['no_fault_mode'] else "Fault Conditions"
    mode_class = "no-fault-mode" if current_mode == "No Faults" else "fault-mode"
    st.markdown(f"""
    <div class="mode-indicator {mode_class}">
        Current Mode: {current_mode}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Fault Injection (only available in fault mode)
    st.markdown("### 🚨 Fault Injection")
    if not simulator.fault_active['no_fault_mode']:
        fault_col1, fault_col2 = st.columns(2)
        with fault_col1:
            if st.button("Vibration Imbalance", key="vibration_btn"):
                simulator.inject_fault("vibration_imbalance")
            if st.button("Bearing Wear", key="bearing_btn"):
                simulator.inject_fault("bearing_wear")
        with fault_col2:
            if st.button("Misalignment", key="misalignment_btn"):
                simulator.inject_fault("misalignment")
            if st.button("🔄 Reset Faults", key="reset_btn"):
                simulator.reset_faults()
    else:
        st.info("Switch to 'Fault Conditions' mode to inject faults")
    
    st.markdown("---")
    
    # Data & Refresh Controls
    st.markdown("### 📊 Data Controls")
    auto_refresh = st.checkbox("Auto Refresh Data", value=True)
    refresh_rate = st.slider("Refresh Rate (sec)", 0.5, 5.0, 1.0, 0.5)
    
    st.markdown("---")
    
    # ML Controls
    st.markdown("### 🤖 ML Controls")
    if st.button("Train Models", key="train_btn"):
        with st.spinner("Training ML models..."):
            predictor.train_models(st.session_state.data)
    
    st.markdown("---")
    
    # Display Options
    st.markdown("### 👁️ Display Options")
    show_raw_data = st.checkbox("Show Raw Data", value=False)
    show_sensor_details = st.checkbox("Show Sensor Details", value=True)

# --- Main Dashboard Logic ---
current_time = datetime.now()
if auto_refresh and (current_time - st.session_state.last_data_gen_time).total_seconds() >= refresh_rate:
    new_data_point = simulator.generate_sensor_data()
    st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_data_point])], ignore_index=True)
    st.session_state.data = st.session_state.data.tail(1000).reset_index(drop=True)
    st.session_state.last_data_gen_time = current_time
    st.rerun()

df = st.session_state.data

if df.empty:
    st.warning("🔄 Starting engine simulation... Please wait for data...")
    time.sleep(refresh_rate)
    st.rerun()

# Auto-train ML models
if not predictor.models_trained and len(df) >= MIN_TRAINING_SAMPLES:
    if (current_time - st.session_state.last_ml_train_time).total_seconds() >= 20:
        with st.spinner("🤖 Auto-training ML models..."):
            predictor.train_models(df)
        if predictor.models_trained:
            st.success("✅ ML models auto-trained successfully!")
        st.session_state.last_ml_train_time = current_time

latest = df.iloc[-1]
df_display = df.tail(200).copy()
df_display['timestamp'] = pd.to_datetime(df_display['timestamp'])

# --- Enhanced Metrics Display ---
st.markdown("### 📊 System Overview")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    health = latest['health_status'].lower()
    card_class = f"{health}-card"
    st.markdown(f"""
    <div class="metric-card {card_class}">
        <h4>Overall Health</h4>
        <h2>{latest['health_status']}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    prev_rpm = df.iloc[-2]['engine_rpm'] if len(df) > 1 else latest['engine_rpm']
    st.metric("Engine RPM", f"{latest['engine_rpm']:.0f}", 
              delta=f"{latest['engine_rpm'] - prev_rpm:.0f}")

with col3:
    prev_rms = df.iloc[-2]['vibration_rms'] if len(df) > 1 else latest['vibration_rms']
    st.metric("Vibration RMS", f"{latest['vibration_rms']:.3f} g", 
              delta=f"{latest['vibration_rms'] - prev_rms:.3f}")

with col4:
    prev_temp = df.iloc[-2]['engine_temp'] if len(df) > 1 else latest['engine_temp']
    st.metric("Temperature", f"{latest['engine_temp']:.1f}°C", 
              delta=f"{latest['engine_temp'] - prev_temp:.1f}")

with col5:
    st.metric("Anomaly Score", f"{latest['anomaly_score']:.1f}", 
              delta=None)

# --- Sensor-wise Status Display ---
if show_sensor_details:
    st.markdown("### 🔍 Individual Sensor Status")
    sensor_col1, sensor_col2, sensor_col3, sensor_col4 = st.columns(4)
    
    sensors = [
        ("Accelerometer", "accelerometer", f"RMS: {latest['vibration_rms']:.3f}g"),
        ("Gyroscope", "gyroscope", f"Tilt: {latest['tilt_angle']:.1f}°"),
        ("Temperature", "temperature", f"Temp: {latest['engine_temp']:.1f}°C"),
        ("RPM Sensor", "rpm", f"RPM: {latest['engine_rpm']:.0f}")
    ]
    
    for i, (sensor_name, sensor_key, sensor_value) in enumerate(sensors):
        with [sensor_col1, sensor_col2, sensor_col3, sensor_col4][i]:
            status = simulator.sensor_status[sensor_key]['status'].lower()
            status_class = f"sensor-{status}"
            st.markdown(f"""
            <div class="sensor-status {status_class}">
                <strong>{sensor_name}</strong><br>
                Status: {simulator.sensor_status[sensor_key]['status']}<br>
                <small>{sensor_value}</small>
            </div>
            """, unsafe_allow_html=True)

# --- ML Predictions ---
st.markdown("### 🤖 ML Predictions")
ml_pred = predictor.predict(df)

if ml_pred:
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        st.info(f"**Health Classification**\n\n"
                f"Prediction: **{ml_pred['health_prediction']}**\n\n"
                f"Confidence: **{ml_pred['health_confidence']:.1f}%**")
    
    with pred_col2:
        if ml_pred['anomaly_detected']:
            st.error("**Anomaly Detection**\n\n🚨 ANOMALY DETECTED")
        else:
            st.success("**Anomaly Detection**\n\n✅ NORMAL OPERATION")
    
    with pred_col3:
        st.info(f"**RUL Estimation**\n\n"
                f"Remaining Life: **{ml_pred['rul_prediction']} hours**")
else:
    if len(df) < MIN_TRAINING_SAMPLES:
        st.info(f"🤖 ML models need more data: {len(df)}/{MIN_TRAINING_SAMPLES} samples collected")
    else:
        st.warning("🤖 ML models not trained yet. Click 'Train Models' in sidebar.")

# --- Sensor-wise Data Visualization ---
st.markdown("### 📈 Real-time Sensor Monitoring")

# Create tabs for different sensor categories
sensor_tab1, sensor_tab2, sensor_tab3, sensor_tab4 = st.tabs(["🔄 Accelerometer", "🌀 Gyroscope", "🌡️ Temperature", "⚙️ Engine Parameters"])

with sensor_tab1:
    st.markdown("#### Accelerometer Data")
    accel_col1, accel_col2 = st.columns(2)
    
    with accel_col1:
        # Accelerometer 3D plot
        fig_accel = go.Figure()
        
        fig_accel.add_trace(go.Scatter(
            x=df_display.index, y=df_display['accel_x'],
            mode='lines', name='X-axis', line=dict(color='red', width=2)
        ))
        fig_accel.add_trace(go.Scatter(
            x=df_display.index, y=df_display['accel_y'],
            mode='lines', name='Y-axis', line=dict(color='green', width=2)
        ))
        fig_accel.add_trace(go.Scatter(
            x=df_display.index, y=df_display['accel_z'],
            mode='lines', name='Z-axis', line=dict(color='blue', width=2)
        ))
        
        fig_accel.update_layout(
            title="Accelerometer Readings (3-Axis)",
            xaxis_title="Sample Index",
            yaxis_title="Acceleration (g)",
            height=400,
            showlegend=True,
            template="plotly_white"
        )
        st.plotly_chart(fig_accel, use_container_width=True)
    
    with accel_col2:
        # Vibration RMS and Magnitude
        fig_vib = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Vibration RMS', 'Acceleration Magnitude'),
            vertical_spacing=0.1
        )
        
        fig_vib.add_trace(
            go.Scatter(x=df_display.index, y=df_display['vibration_rms'],
                      mode='lines', name='RMS', line=dict(color='purple', width=2)),
            row=1, col=1
        )
        
        fig_vib.add_trace(
            go.Scatter(x=df_display.index, y=df_display['accel_magnitude'],
                      mode='lines', name='Magnitude', line=dict(color='orange', width=2)),
            row=2, col=1
        )
        
        fig_vib.update_layout(height=400, showlegend=False, template="plotly_white")
        fig_vib.update_xaxes(title_text="Sample Index", row=2, col=1)
        fig_vib.update_yaxes(title_text="RMS (g)", row=1, col=1)
        fig_vib.update_yaxes(title_text="Magnitude (g)", row=2, col=1)
        
        st.plotly_chart(fig_vib, use_container_width=True)
    
    # Accelerometer status and statistics
    st.markdown("**Current Accelerometer Status:**")
    accel_status_col1, accel_status_col2, accel_status_col3 = st.columns(3)
    with accel_status_col1:
        st.metric("X-axis", f"{latest['accel_x']:.3f} g")
    with accel_status_col2:
        st.metric("Y-axis", f"{latest['accel_y']:.3f} g")
    with accel_status_col3:
        st.metric("Z-axis", f"{latest['accel_z']:.3f} g")

with sensor_tab2:
    st.markdown("#### Gyroscope Data")
    gyro_col1, gyro_col2 = st.columns(2)
    
    with gyro_col1:
        # Gyroscope 3D plot
        fig_gyro = go.Figure()
        
        fig_gyro.add_trace(go.Scatter(
            x=df_display.index, y=df_display['gyro_x'],
            mode='lines', name='X-axis', line=dict(color='red', width=2)
        ))
        fig_gyro.add_trace(go.Scatter(
            x=df_display.index, y=df_display['gyro_y'],
            mode='lines', name='Y-axis', line=dict(color='green', width=2)
        ))
        fig_gyro.add_trace(go.Scatter(
            x=df_display.index, y=df_display['gyro_z'],
            mode='lines', name='Z-axis', line=dict(color='blue', width=2)
        ))
        
        fig_gyro.update_layout(
            title="Gyroscope Readings (3-Axis)",
            xaxis_title="Sample Index",
            yaxis_title="Angular Velocity (°/s)",
            height=400,
            showlegend=True,
            template="plotly_white"
        )
        st.plotly_chart(fig_gyro, use_container_width=True)
    
    with gyro_col2:
        # Tilt angle and gyroscope magnitude
        gyro_magnitude = np.sqrt(df_display['gyro_x']**2 + df_display['gyro_y']**2 + df_display['gyro_z']**2)
        
        fig_tilt = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Tilt Angle', 'Gyroscope Magnitude'),
            vertical_spacing=0.1
        )
        
        fig_tilt.add_trace(
            go.Scatter(x=df_display.index, y=df_display['tilt_angle'],
                      mode='lines', name='Tilt', line=dict(color='cyan', width=2)),
            row=1, col=1
        )
        
        fig_tilt.add_trace(
            go.Scatter(x=df_display.index, y=gyro_magnitude,
                      mode='lines', name='Magnitude', line=dict(color='magenta', width=2)),
            row=2, col=1
        )
        
        fig_tilt.update_layout(height=400, showlegend=False, template="plotly_white")
        fig_tilt.update_xaxes(title_text="Sample Index", row=2, col=1)
        fig_tilt.update_yaxes(title_text="Angle (°)", row=1, col=1)
        fig_tilt.update_yaxes(title_text="Magnitude (°/s)", row=2, col=1)
        
        st.plotly_chart(fig_tilt, use_container_width=True)
    
    # Gyroscope status and statistics
    st.markdown("**Current Gyroscope Status:**")
    gyro_status_col1, gyro_status_col2, gyro_status_col3 = st.columns(3)
    with gyro_status_col1:
        st.metric("X-axis", f"{latest['gyro_x']:.3f} °/s")
    with gyro_status_col2:
        st.metric("Y-axis", f"{latest['gyro_y']:.3f} °/s")
    with gyro_status_col3:
        st.metric("Z-axis", f"{latest['gyro_z']:.3f} °/s")

with sensor_tab3:
    st.markdown("#### Temperature Monitoring")
    temp_col1, temp_col2 = st.columns(2)
    
    with temp_col1:
        # Temperature trend
        fig_temp = go.Figure()
        
        # Add temperature zones
        fig_temp.add_hline(y=85, line_dash="dash", line_color="green", 
                          annotation_text="Normal Range", annotation_position="bottom right")
        fig_temp.add_hline(y=95, line_dash="dash", line_color="orange", 
                          annotation_text="Warning Threshold", annotation_position="bottom right")
        fig_temp.add_hline(y=105, line_dash="dash", line_color="red", 
                          annotation_text="Fault Threshold", annotation_position="bottom right")
        
        fig_temp.add_trace(go.Scatter(
            x=df_display.index, y=df_display['engine_temp'],
            mode='lines+markers', name='Temperature',
            line=dict(color='red', width=3),
            marker=dict(size=4)
        ))
        
        fig_temp.update_layout(
            title="Engine Temperature Trend",
            xaxis_title="Sample Index",
            yaxis_title="Temperature (°C)",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with temp_col2:
        # Temperature distribution and statistics
        temp_stats = df_display['engine_temp'].describe()
        
        fig_temp_hist = go.Figure()
        fig_temp_hist.add_trace(go.Histogram(
            x=df_display['engine_temp'],
            nbinsx=20,
            name='Temperature Distribution',
            marker_color='red',
            opacity=0.7
        ))
        
        fig_temp_hist.update_layout(
            title="Temperature Distribution",
            xaxis_title="Temperature (°C)",
            yaxis_title="Frequency",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_temp_hist, use_container_width=True)
    
    # Temperature statistics
    st.markdown("**Temperature Statistics:**")
    temp_stat_col1, temp_stat_col2, temp_stat_col3, temp_stat_col4 = st.columns(4)
    with temp_stat_col1:
        st.metric("Current", f"{latest['engine_temp']:.1f} °C")
    with temp_stat_col2:
        st.metric("Average", f"{temp_stats['mean']:.1f} °C")
    with temp_stat_col3:
        st.metric("Maximum", f"{temp_stats['max']:.1f} °C")
    with temp_stat_col4:
        st.metric("Minimum", f"{temp_stats['min']:.1f} °C")

with sensor_tab4:
    st.markdown("#### Engine Parameters")
    engine_col1, engine_col2 = st.columns(2)
    
    with engine_col1:
        # RPM and Load
        fig_engine = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Engine RPM', 'Engine Load'),
            vertical_spacing=0.1
        )
        
        fig_engine.add_trace(
            go.Scatter(x=df_display.index, y=df_display['engine_rpm'],
                      mode='lines', name='RPM', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig_engine.add_trace(
            go.Scatter(x=df_display.index, y=df_display['engine_load'],
                      mode='lines', name='Load', line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        fig_engine.update_layout(height=400, showlegend=False, template="plotly_white")
        fig_engine.update_xaxes(title_text="Sample Index", row=2, col=1)
        fig_engine.update_yaxes(title_text="RPM", row=1, col=1)
        fig_engine.update_yaxes(title_text="Load Factor", row=2, col=1)
        
        st.plotly_chart(fig_engine, use_container_width=True)
    
    with engine_col2:
        # Operating state distribution and anomaly score
        fig_state = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Anomaly Score Trend', 'Stability Index'),
            vertical_spacing=0.1
        )
        
        fig_state.add_trace(
            go.Scatter(x=df_display.index, y=df_display['anomaly_score'],
                      mode='lines', name='Anomaly Score', 
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        fig_state.add_trace(
            go.Scatter(x=df_display.index, y=df_display['stability_index'],
                      mode='lines', name='Stability Index', 
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        fig_state.update_layout(height=400, showlegend=False, template="plotly_white")
        fig_state.update_xaxes(title_text="Sample Index", row=2, col=1)
        fig_state.update_yaxes(title_text="Score", row=1, col=1)
        fig_state.update_yaxes(title_text="Index", row=2, col=1)
        
        st.plotly_chart(fig_state, use_container_width=True)
    
    # Engine parameter statistics
    st.markdown("**Engine Parameter Status:**")
    engine_stat_col1, engine_stat_col2, engine_stat_col3, engine_stat_col4 = st.columns(4)
    with engine_stat_col1:
        st.metric("Current RPM", f"{latest['engine_rpm']:.0f}")
    with engine_stat_col2:
        st.metric("Load Factor", f"{latest['engine_load']:.2f}")
    with engine_stat_col3:
        st.metric("Operating State", latest['operating_state'].title())
    with engine_stat_col4:
        st.metric("Stability Index", f"{latest['stability_index']:.1f}%")

# --- System Health Summary ---
st.markdown("### 🏥 System Health Summary")

health_col1, health_col2 = st.columns(2)

with health_col1:
    # Health status over time
    health_numeric = df_display['health_status'].map({'Normal': 0, 'Warning': 1, 'Fault': 2})
    
    fig_health = go.Figure()
    fig_health.add_trace(go.Scatter(
        x=df_display.index, y=health_numeric,
        mode='lines+markers',
        name='Health Status',
        line=dict(color='purple', width=3),
        marker=dict(size=6)
    ))
    
    fig_health.update_layout(
        title="Health Status Timeline",
        xaxis_title="Sample Index",
        yaxis_title="Health Level",
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=['Normal', 'Warning', 'Fault']
        ),
        height=300,
        template="plotly_white"
    )
    st.plotly_chart(fig_health, use_container_width=True)

with health_col2:
    # Health distribution pie chart
    health_counts = df_display['health_status'].value_counts()
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=health_counts.index,
        values=health_counts.values,
        hole=0.4,
        marker_colors=['green', 'orange', 'red']
    )])
    
    fig_pie.update_layout(
        title="Health Status Distribution",
        height=300,
        template="plotly_white"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Active Faults Display ---
st.markdown("### 🚨 Active Fault Conditions")
active_faults = [fault for fault, status in simulator.fault_active.items() 
                if status and fault != 'no_fault_mode']

if active_faults:
    fault_display_col1, fault_display_col2 = st.columns(2)
    
    with fault_display_col1:
        st.error("**Currently Active Faults:**")
        for fault in active_faults:
            fault_display_names = {
                'vibration_imbalance': '🔄 Vibration Imbalance',
                'bearing_wear': '🔧 Bearing Wear',
                'misalignment': '⚖️ Shaft Misalignment'
            }
            st.write(f"• {fault_display_names.get(fault, fault)}")
    
    with fault_display_col2:
        st.info("**Fault Impact on Sensors:**")
        if 'vibration_imbalance' in active_faults:
            st.write("• Accelerometer: High RMS values")
        if 'bearing_wear' in active_faults:
            st.write("• Accelerometer: Increased noise")
            st.write("• Gyroscope: Higher angular velocity")
        if 'misalignment' in active_faults:
            st.write("• Gyroscope: Significant deviation")
            st.write("• Vibration: Increased skewness")
else:
    st.success("✅ **No Active Faults** - System operating normally")

# --- Data Export and Raw Data Display ---
if show_raw_data:
    st.markdown("### 📋 Raw Sensor Data")
    
    # Export functionality
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        if st.button("📥 Export Data to CSV"):
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"engine_sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with export_col2:
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Data Collection Started:** {df.iloc[0]['timestamp'] if len(df) > 0 else 'N/A'}")
    
    # Display recent data
    st.dataframe(
        df.tail(20)[['timestamp', 'health_status', 'engine_rpm', 'engine_temp', 
                    'vibration_rms', 'anomaly_score', 'operating_state']],
        use_container_width=True
    )

# --- Footer with system information ---
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**📊 Data Statistics**")
    st.write(f"Records: {len(df)}")
    st.write(f"Update Rate: {refresh_rate}s")

with footer_col2:
    st.markdown("**🤖 ML Status**")
    ml_status = "✅ Trained" if predictor.models_trained else "❌ Not Trained"
    st.write(f"Models: {ml_status}")
    st.write(f"Min Samples: {MIN_TRAINING_SAMPLES}")

with footer_col3:
    st.markdown("**⏱️ System Status**")
    st.write(f"Last Update: {latest['timestamp']}")
    st.write(f"Auto Refresh: {'✅ On' if auto_refresh else '❌ Off'}")

# Auto-refresh mechanism using streamlit-autorefresh
if auto_refresh and st_autorefresh:
    refresh_interval_ms = int(refresh_rate * 1000)  # Convert to milliseconds
    st_autorefresh(interval=refresh_interval_ms, key="vibration_autorefresh")