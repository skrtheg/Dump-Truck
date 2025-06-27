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

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global constants
MIN_TRAINING_SAMPLES = 50

class EngineComponentSimulator:
    """
    Enhanced engine sensor simulator with realistic operation profiles and proper fault injection
    """
    def __init__(self):
        self.time_step = 0
        
        # More realistic engine states with proper parameter ranges
        self.engine_states = {
            'idle': {
                'rpm': (700, 900), 
                'load': (0.02, 0.08), 
                'temp': (70, 80),
                'base_vibration': 0.08,  # Very low vibration at idle
                'base_rotation': 0.3
            },
            'light_load': {
                'rpm': (900, 1200), 
                'load': (0.15, 0.35), 
                'temp': (78, 88),
                'base_vibration': 0.12,
                'base_rotation': 0.8
            },
            'normal_load': {
                'rpm': (1200, 1600), 
                'load': (0.35, 0.65), 
                'temp': (85, 95),
                'base_vibration': 0.18,
                'base_rotation': 1.2
            },
            'heavy_load': {
                'rpm': (1600, 2000), 
                'load': (0.65, 0.85), 
                'temp': (92, 105),
                'base_vibration': 0.25,
                'base_rotation': 1.8
            },
            'max_load': {
                'rpm': (2000, 2400), 
                'load': (0.85, 0.98), 
                'temp': (100, 120),
                'base_vibration': 0.35,
                'base_rotation': 2.5
            }
        }
        
        # Fault management system
        self.fault_states = {
            'no_fault': False,
            'vibration_imbalance': False,
            'bearing_wear': False,
            'misalignment': False,
            'temperature_spike': False,
            'rpm_instability': False
        }
        
        # Reduced base noise levels for more realistic sensor behavior
        self.sensor_noise = {
            'rpm': 3.0,      # Reduced from 5.0
            'load': 0.01,    # Reduced from 0.02
            'temp': 0.8,     # Reduced from 1.5
            'accel': 0.02,   # Reduced from 0.05
            'gyro': 0.1      # Reduced from 0.2
        }

        # Adjusted anomaly thresholds for proper 'Normal' operation
        self.anomaly_thresholds = {
            'vibration_rms_threshold': 0.2,      # Adjusted: Lower for dynamic vibration after subtracting gravity
            'temp_threshold': 110,               
            'rpm_deviation_threshold': 250,      
            'gyro_magnitude_threshold': 4.0,     
            'tilt_angle_threshold': 20,          
            'vibration_skew_threshold': 1.5,     
            'caution_score': 20,                 
            'warning_score': 40,
            'critical_score': 70,
            'base_rul': 10000,
            'rul_impact_factor': 60              
        }

    def set_fault_mode(self, fault_type: str):
        """Set specific fault mode, disabling others"""
        for fault in self.fault_states:
            self.fault_states[fault] = False
        
        if fault_type in self.fault_states:
            self.fault_states[fault_type] = True
            
        return fault_type

    def get_current_fault_status(self):
        """Return current active fault"""
        for fault, active in self.fault_states.items():
            if active:
                return fault
        return 'no_fault'

    def update_anomaly_thresholds(self, new_thresholds: dict):
        """Update anomaly scoring thresholds based on user input"""
        self.anomaly_thresholds.update(new_thresholds)

    def generate_realistic_operating_state(self):
        """Generate realistic operating state based on time and load patterns"""
        states = ['idle', 'light_load', 'normal_load', 'heavy_load', 'max_load']
        
        if self.time_step < 30:
            probs = [0.7, 0.25, 0.05, 0.0, 0.0]
        elif self.time_step % 120 < 15:
            probs = [0.6, 0.3, 0.1, 0.0, 0.0]
        else:
            probs = [0.05, 0.4, 0.45, 0.08, 0.02]
            
        return np.random.choice(states, p=probs)

    def apply_fault_effects(self, base_data: dict, fault_type: str) -> dict:
        """Apply specific fault effects to sensor data with more realistic magnitudes"""
        data = base_data.copy()
        
        if fault_type == 'vibration_imbalance':
            data['accel_x'] += np.random.normal(0.8, 0.3)
            data['accel_y'] += np.random.normal(0.7, 0.25)
            data['accel_z'] += np.random.normal(0.5, 0.2)
            data['gyro_x'] += np.random.normal(2.0, 0.8)
            data['gyro_y'] += np.random.normal(1.8, 0.7)
            
        elif fault_type == 'bearing_wear':
            data['accel_x'] += np.random.normal(0, 0.4) + 0.3 * np.sin(self.time_step * 0.5)
            data['accel_y'] += np.random.normal(0, 0.35) + 0.25 * np.cos(self.time_step * 0.7)
            data['accel_z'] += np.random.normal(0, 0.3) + 0.2 * np.sin(self.time_step * 0.9)
            data['engine_temp'] += np.random.uniform(3, 8)
            
        elif fault_type == 'misalignment':
            data['accel_x'] += 0.6 * np.sin(self.time_step * 0.3)
            data['accel_y'] += 0.5 * np.cos(self.time_step * 0.4)
            data['gyro_z'] += np.random.normal(1.5, 0.5)
            data['engine_rpm'] += np.random.normal(0, 30)
            
        elif fault_type == 'temperature_spike':
            temp_spike = np.random.uniform(8, 20)
            data['engine_temp'] += temp_spike
            if temp_spike > 15:
                data['engine_load'] = min(1.0, data['engine_load'] + 0.05)
                
        elif fault_type == 'rpm_instability':
            rpm_variation = np.random.normal(0, 80)
            data['engine_rpm'] = max(500, data['engine_rpm'] + rpm_variation)
            data['gyro_x'] += rpm_variation * 0.002
            data['gyro_y'] += rpm_variation * 0.0015
            
        return data

    def generate_sensor_data(self) -> dict:
        """Generate comprehensive sensor data with realistic distributions"""
        self.time_step += 1
        
        operating_state = self.generate_realistic_operating_state()
        state_params = self.engine_states[operating_state]
        
        engine_rpm = np.random.uniform(*state_params['rpm'])
        engine_load = np.random.uniform(*state_params['load'])
        engine_temp = np.random.uniform(*state_params['temp'])
        
        engine_rpm += np.random.normal(0, self.sensor_noise['rpm'])
        engine_load += np.random.normal(0, self.sensor_noise['load'])
        engine_temp += np.random.normal(0, self.sensor_noise['temp'])
        
        engine_rpm = max(500, engine_rpm)
        engine_load = np.clip(engine_load, 0, 1)
        engine_temp = max(50, engine_temp)
        
        base_vibration = state_params['base_vibration']
        accel_x = np.random.normal(0, base_vibration + self.sensor_noise['accel'])
        accel_y = np.random.normal(0, base_vibration + self.sensor_noise['accel'])
        accel_z_dynamic = np.random.normal(0, base_vibration * 0.3 + self.sensor_noise['accel'])
        accel_z = 9.81 + accel_z_dynamic
        
        base_rotation = state_params['base_rotation']
        gyro_x = np.random.normal(0, base_rotation + self.sensor_noise['gyro'])
        gyro_y = np.random.normal(0, base_rotation * 0.8 + self.sensor_noise['gyro'])
        gyro_z = np.random.normal(0, base_rotation * 0.6 + self.sensor_noise['gyro'])
        
        base_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'engine_rpm': round(engine_rpm, 1),
            'engine_load': round(engine_load, 3),
            'engine_temp': round(engine_temp, 2),
            'accel_x': round(accel_x, 4),
            'accel_y': round(accel_y, 4),
            'accel_z': round(accel_z, 4),
            'gyro_x': round(gyro_x, 4),
            'gyro_y': round(gyro_y, 4),
            'gyro_z': round(gyro_z, 4),
            'operating_state': operating_state
        }
        
        current_fault = self.get_current_fault_status()
        if current_fault != 'no_fault':
            base_data = self.apply_fault_effects(base_data, current_fault)
        
        current_accel_x = base_data['accel_x']
        current_accel_y = base_data['accel_y']
        current_accel_z = base_data['accel_z']

        accel_x_dynamic = current_accel_x
        accel_y_dynamic = current_accel_y
        accel_z_dynamic = current_accel_z - 9.81

        vibration_rms = np.sqrt(np.mean([accel_x_dynamic**2, accel_y_dynamic**2, accel_z_dynamic**2]))
        
        accel_mag = np.sqrt(current_accel_x**2 + current_accel_y**2 + current_accel_z**2)
        gyro_mag = np.sqrt(base_data['gyro_x']**2 + base_data['gyro_y']**2 + base_data['gyro_z']**2)
        tilt_angle = np.degrees(np.arctan2(np.sqrt(current_accel_x**2 + current_accel_y**2), abs(current_accel_z)))
        
        if current_fault == 'no_fault':
            vibration_skew = np.random.normal(0, 0.2)
        else:
            if current_fault == 'vibration_imbalance' or current_fault == 'bearing_wear':
                vibration_skew = np.random.uniform(1.0, 3.0)
            else:
                vibration_skew = np.random.uniform(0.5, 1.5)
            
            if np.random.rand() > 0.5:
                vibration_skew = -vibration_skew
            
        anomaly_score = 0
        
        if vibration_rms > self.anomaly_thresholds['vibration_rms_threshold']:
            anomaly_score += (vibration_rms - self.anomaly_thresholds['vibration_rms_threshold']) * 15 
        
        if base_data['engine_temp'] > self.anomaly_thresholds['temp_threshold']:
            anomaly_score += (base_data['engine_temp'] - self.anomaly_thresholds['temp_threshold']) * 1.5
        
        expected_rpm_mid = np.mean(state_params['rpm'])
        rpm_deviation = abs(base_data['engine_rpm'] - expected_rpm_mid)
        if rpm_deviation > self.anomaly_thresholds['rpm_deviation_threshold']:
            anomaly_score += (rpm_deviation - self.anomaly_thresholds['rpm_deviation_threshold']) * 0.1
        
        if gyro_mag > self.anomaly_thresholds['gyro_magnitude_threshold']:
            anomaly_score += (gyro_mag - self.anomaly_thresholds['gyro_magnitude_threshold']) * 8
        
        if tilt_angle > self.anomaly_thresholds['tilt_angle_threshold']:
            anomaly_score += (tilt_angle - self.anomaly_thresholds['tilt_angle_threshold']) * 2
        
        if abs(vibration_skew) > self.anomaly_thresholds['vibration_skew_threshold']:
            anomaly_score += abs(vibration_skew) * 8
        
        anomaly_score = min(100, max(0, anomaly_score))
        
        if anomaly_score > self.anomaly_thresholds['critical_score']:
            health = 'Critical'
        elif anomaly_score > self.anomaly_thresholds['warning_score']:
            health = 'Warning'
        elif anomaly_score > self.anomaly_thresholds['caution_score']:
            health = 'Caution'
        else:
            health = 'Normal'
            
        base_rul = self.anomaly_thresholds['base_rul']
        rul = max(0, base_rul - anomaly_score * self.anomaly_thresholds['rul_impact_factor'])
        
        stability = max(0, 100 - anomaly_score * 1.1)
        
        base_data.update({
            'accel_magnitude': round(accel_mag, 4),
            'vibration_rms': round(vibration_rms, 4),
            'gyro_magnitude': round(gyro_mag, 4),
            'tilt_angle': round(tilt_angle, 2),
            'vibration_skew': round(vibration_skew, 3),
            'stability_index': round(stability, 2),
            'anomaly_score': round(anomaly_score, 2),
            'health_status': health,
            'rul_estimate': int(rul),
            'fault_status': current_fault
        })
        
        return base_data

class MLPredictor:
    """Enhanced ML predictor with better feature handling"""
    def __init__(self):
        self.svm_model = None
        self.dt_model = None
        self.scaler = StandardScaler()
        self.models_trained = False
        self.health_labels = ['Normal', 'Caution', 'Warning', 'Critical']
        self.health_map = {label: i for i, label in enumerate(self.health_labels)}
        # Store the classes the SVM model was trained on
        self.svm_trained_classes = None 

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        feature_cols = [
            'engine_rpm', 'engine_load', 'engine_temp',
            'accel_x', 'accel_y', 'accel_z', 'accel_magnitude',
            'gyro_x', 'gyro_y', 'gyro_z', 'gyro_magnitude',
            'vibration_rms', 'vibration_skew', 'tilt_angle'
        ]
        
        X = df[feature_cols].copy()
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
        return X

    def train_models(self, df: pd.DataFrame) -> bool:
        """Train ML models with enhanced error handling"""
        if len(df) < MIN_TRAINING_SAMPLES:
            st.warning(f"Not enough data for ML training. Need at least {MIN_TRAINING_SAMPLES} samples, but have {len(df)}.")
            self.models_trained = False
            return False
        
        try:
            X = self.prepare_features(df)
            y_health = df['health_status'].map(self.health_map).fillna(0)
            y_anomaly = (df['anomaly_score'] > 15).astype(int)

            if len(np.unique(y_health)) < 2:
                st.warning("Not enough unique health statuses for SVM training. Need at least two health categories (e.g., Normal and Caution/Warning).")
                self.models_trained = False
                return False
            if len(np.unique(y_anomaly)) < 2:
                st.warning("Not enough unique anomaly statuses for Decision Tree training. Need both 'normal' and 'anomalous' examples.")
                self.models_trained = False
                return False
            
            X_scaled = self.scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_health, test_size=0.2, random_state=42, stratify=y_health
            )
            
            self.svm_model = SVC(kernel='rbf', probability=True, random_state=42, C=0.8)
            self.svm_model.fit(X_train, y_train)
            # Store the classes that the SVM model was trained with
            self.svm_trained_classes = self.svm_model.classes_ 
            
            X_train_anom, X_test_anom, y_train_anom, y_test_anom = train_test_split(
                X_scaled, y_anomaly, test_size=0.2, random_state=42, stratify=y_anomaly
            )
            
            self.dt_model = DecisionTreeClassifier(random_state=42, max_depth=8, min_samples_split=5)
            self.dt_model.fit(X_train_anom, y_train_anom)
            
            self.models_trained = True
            return True
            
        except Exception as e:
            st.error(f"ML Training Error: {e}")
            self.models_trained = False
            return False
    
    def predict(self, df_latest: pd.DataFrame) -> dict:
        """Make predictions on latest data"""
        if not self.models_trained or df_latest.empty:
            return None
        
        try:
            latest_features = self.prepare_features(df_latest.tail(1))
            X_scaled_latest = self.scaler.transform(latest_features)
            
            health_pred_num = self.svm_model.predict(X_scaled_latest)[0]
            
            # Get raw probabilities for the classes the model knows
            raw_health_proba = self.svm_model.predict_proba(X_scaled_latest)[0]
            
            # Map raw probabilities to the full set of health labels
            full_health_proba = np.zeros(len(self.health_labels))
            for i, class_val in enumerate(self.svm_trained_classes):
                # Map the trained class (numerical) back to its index in self.health_labels
                # This assumes self.health_labels is ordered (Normal, Caution, Warning, Critical)
                # and self.health_map reflects this order.
                label_index = list(self.health_map.keys())[list(self.health_map.values()).index(class_val)]
                full_health_proba[self.health_labels.index(label_index)] = raw_health_proba[i]

            anomaly_pred_num = self.dt_model.predict(X_scaled_latest)[0]
            rul_pred = df_latest.iloc[-1]['rul_estimate']
            
            return {
                'health_prediction': self.health_labels[int(health_pred_num)],
                'health_confidence': max(raw_health_proba) * 100, # Max confidence is from the predicted class
                'anomaly_detected': bool(anomaly_pred_num),
                'rul_prediction': int(rul_pred),
                'full_health_probabilities': full_health_proba # Return full array for charting
            }
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            return None

# Initialize session state
if 'simulator' not in st.session_state:
    st.session_state.simulator = EngineComponentSimulator()
    st.session_state.predictor = MLPredictor()
    st.session_state.data = pd.DataFrame()
    st.session_state.last_data_gen_time = datetime.now()
    st.session_state.last_ml_train_time = datetime.now()
    st.session_state.anomaly_thresholds = st.session_state.simulator.anomaly_thresholds.copy()

simulator = st.session_state.simulator
predictor = st.session_state.predictor

simulator.update_anomaly_thresholds(st.session_state.anomaly_thresholds)

# Custom CSS with improved styling (kept the same as previous successful version)
st.markdown("""
<style>
    .metric-card {
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .normal-card { background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); }
    .caution-card { background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); }
    .warning-card { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }
    .critical-card { background: linear-gradient(135deg, #8e44ad 0%, #732d91 100%); }
    .sensor-card {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .fault-active {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    .fault-inactive {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .stMetric > div > div > div > div {
        color: #2c3e50;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔧 Advanced Engine Component Monitoring System")
st.markdown("**Real-time multi-sensor monitoring with intelligent fault detection and ML analytics**")

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ System Controls")
    
    st.markdown("### 📊 Data Collection")
    auto_refresh = st.checkbox("Auto Refresh Data", value=True)
    refresh_rate = st.slider("Refresh Rate (seconds)", 0.5, 5.0, 1.0, 0.5)
    
    st.markdown("---")
    st.markdown("### 🚨 Fault Injection System")
    
    current_fault = simulator.get_current_fault_status()
    st.markdown(f"""
    <div class="{'fault-active' if current_fault != 'no_fault' else 'fault-inactive'}">
        Current Status: <strong>{current_fault.replace('_', ' ').title()}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    fault_options = {
        'no_fault': '✅ Normal Operation',
        'vibration_imbalance': '⚡ Vibration Imbalance',
        'bearing_wear': '🔧 Bearing Wear',
        'misalignment': '📐 Shaft Misalignment',
        'temperature_spike': '🌡️ Temperature Spike',
        'rpm_instability': '📊 RPM Instability'
    }
    
    selected_fault = st.selectbox(
        "Select Fault Condition",
        options=list(fault_options.keys()),
        format_func=lambda x: fault_options[x],
        index=list(fault_options.keys()).index(current_fault)
    )
    
    if st.button("Apply Fault Condition", type="primary"):
        new_fault = simulator.set_fault_mode(selected_fault)
        st.success(f"Applied: {fault_options[new_fault]}")
        time.sleep(0.5)
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🤖 ML Model Controls")
    
    if st.button("🏋️ Train ML Models", type="secondary"):
        with st.spinner("Training ML models..."):
            success = predictor.train_models(st.session_state.data)
        if success:
            st.success("✅ Models trained successfully!")
        else:
            st.warning("⚠️ Insufficient data or class imbalance for training.")
    
    if len(st.session_state.data) > 0:
        st.info(f"📈 Data points collected: {len(st.session_state.data)}")
        st.info(f"🎯 ML Model status: {'Trained' if predictor.models_trained else 'Not trained'}")
    
    st.markdown("---")
    show_raw_data = st.checkbox("Show Raw Data", value=False)

# Data generation logic
current_time = datetime.now()
if auto_refresh and (current_time - st.session_state.last_data_gen_time).total_seconds() >= refresh_rate:
    new_data_point = simulator.generate_sensor_data()
    st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_data_point])], ignore_index=True)
    st.session_state.data = st.session_state.data.tail(1000).reset_index(drop=True)
    st.session_state.last_data_gen_time = current_time
    st.rerun()

df = st.session_state.data

if df.empty:
    st.warning("🔄 Initializing engine monitoring system... Please wait for data collection...")
    time.sleep(refresh_rate)
    st.rerun()

# Auto-train ML models
if not predictor.models_trained and len(df) >= MIN_TRAINING_SAMPLES:
    if (current_time - st.session_state.last_ml_train_time).total_seconds() >= 30:
        with st.spinner("🤖 Auto-training ML models..."):
            success = predictor.train_models(df)
        if success:
            st.success("✅ ML models auto-trained successfully!")
        st.session_state.last_ml_train_time = current_time

# Prepare display data
latest = df.iloc[-1]
df_display = df.tail(200).copy()
df_display['timestamp'] = pd.to_datetime(df_display['timestamp'])

# Create tabbed interface
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 System Overview",
    "🏭 Engine Core Sensors",    
    "📳 Vibration Sensors",
    "🔄 Rotation Sensors",
    "🤖 ML Analytics",
    "📊 Historical Data",
    "⚙️ Sensor Thresholds"
])

with tab1:  # System Overview
    st.header("System Overview & Health Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        health = latest['health_status'].lower()
        card_class = f"{health}-card"
        st.markdown(f"""
        <div class="metric-card {card_class}">
            <h4>🏥 Health Status</h4>
            <h2>{latest['health_status']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        delta_val = latest['anomaly_score'] - (df.iloc[-2]['anomaly_score'] if len(df) > 1 else latest['anomaly_score'])
        st.metric("🚨 Anomaly Score", f"{latest['anomaly_score']:.1f}/100", delta=f"{delta_val:.1f}")
    
    with col3:
        stability_delta = latest['stability_index'] - (df.iloc[-2]['stability_index'] if len(df) > 1 else latest['stability_index'])
        st.metric("📊 Stability Index", f"{latest['stability_index']:.1f}%", delta=f"{stability_delta:.1f}")
    
    with col4:
        rul_delta = latest['rul_estimate'] - (df.iloc[-2]['rul_estimate'] if len(df) > 1 else latest['rul_estimate'])
        st.metric("⏰ RUL Estimate", f"{latest['rul_estimate']:,} hrs", delta=f"{rul_delta:,}")

    st.markdown("### 🚨 Current System Status")
    fault_status = latest['fault_status']
    if fault_status != 'no_fault':
        st.error(f"⚠️ **FAULT DETECTED**: {fault_status.replace('_', ' ').title()}")
    else:
        st.success("✅ **NORMAL OPERATION**: All systems functioning within parameters")
    
    st.markdown("### 📈 Key Performance Indicators")
    
    overview_col1, overview_col2 = st.columns(2)
    
    with overview_col1:
        fig_anomaly = px.line(df_display, x='timestamp', y='anomaly_score',
                              title='Anomaly Score Trend', color_discrete_sequence=['#e74c3c'])
        fig_anomaly.add_hline(y=simulator.anomaly_thresholds['caution_score'], line_dash="dash", line_color="yellow", annotation_text="Caution Threshold")
        fig_anomaly.add_hline(y=simulator.anomaly_thresholds['warning_score'], line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
        fig_anomaly.add_hline(y=simulator.anomaly_thresholds['critical_score'], line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        fig_anomaly.update_layout(height=350, xaxis_title="Time", yaxis_title="Anomaly Score")
        st.plotly_chart(fig_anomaly, use_container_width=True)
    
    with overview_col2:
        health_counts = df_display['health_status'].value_counts().reindex(predictor.health_labels, fill_value=0)
        fig_health = px.pie(values=health_counts.values, names=health_counts.index,
                            title='Health Status Distribution (Last 200 Samples)',
                            color_discrete_map={
                                'Normal': '#27ae60',
                                'Caution': '#f39c12',
                                'Warning': '#e67e22',
                                'Critical': '#e74c3c'
                            })
        fig_health.update_layout(height=350)
        st.plotly_chart(fig_health, use_container_width=True)

with tab2:  # Engine Core Sensors
    st.header("🏭 Engine Core Performance Sensors")
    st.markdown("Monitor fundamental engine parameters: **RPM**, **Load**, and **Temperature**.")
    
    core_col1, core_col2, core_col3 = st.columns(3)
    
    with core_col1:
        st.markdown("""
        <div class="sensor-card">
            <h4>🔄 Engine RPM</h4>
            <h2>{:.0f} RPM</h2>
            <p>Operating State: {}</p>
        </div>
        """.format(latest['engine_rpm'], latest['operating_state'].replace('_', ' ').title()),
        unsafe_allow_html=True)
        
        fig_rpm = px.line(df_display, x='timestamp', y='engine_rpm',
                          title='Engine RPM Trend', color_discrete_sequence=['#3498db'])
        expected_rpm_mid = np.mean(simulator.engine_states[latest['operating_state']]['rpm'])
        fig_rpm.add_hline(y=expected_rpm_mid + simulator.anomaly_thresholds['rpm_deviation_threshold'],
                          line_dash="dash", line_color="red", annotation_text="Upper Instability")
        fig_rpm.add_hline(y=expected_rpm_mid - simulator.anomaly_thresholds['rpm_deviation_threshold'],
                          line_dash="dash", line_color="red", annotation_text="Lower Instability")
        fig_rpm.update_layout(height=300, showlegend=False, xaxis_title="Time", yaxis_title="RPM")
        st.plotly_chart(fig_rpm, use_container_width=True)
    
    with core_col2:
        st.markdown("""
        <div class="sensor-card">
            <h4>⚡ Engine Load</h4>
            <h2>{:.2f}</h2>
            <p>Load Factor: {:.1f}%</p>
        </div>
        """.format(latest['engine_load'], latest['engine_load'] * 100),
        unsafe_allow_html=True)
        
        fig_load = px.line(df_display, x='timestamp', y='engine_load',
                           title='Engine Load Trend', color_discrete_sequence=['#e67e22'])
        fig_load.update_layout(height=300, showlegend=False, xaxis_title="Time", yaxis_title="Load")
        st.plotly_chart(fig_load, use_container_width=True)
    
    with core_col3:
        st.markdown("""
        <div class="sensor-card">
            <h4>🌡️ Engine Temperature</h4>
            <h2>{:.1f}°C</h2>
            <p>Status: {}</p>
        </div>
        """.format(latest['engine_temp'],
                   'Normal' if latest['engine_temp'] < simulator.anomaly_thresholds['temp_threshold'] else 'Elevated'),
        unsafe_allow_html=True)
        
        fig_temp = px.line(df_display, x='timestamp', y='engine_temp',
                           title='Engine Temperature Trend (°C)', color_discrete_sequence=['#e74c3c'])
        fig_temp.add_hline(y=simulator.anomaly_thresholds['temp_threshold'], line_dash="dash", line_color="red", annotation_text="High Temp Threshold")
        fig_temp.update_layout(height=300, showlegend=False, xaxis_title="Time", yaxis_title="Temperature (°C)")
        st.plotly_chart(fig_temp, use_container_width=True)

with tab3: # Vibration Sensors
    st.header("📳 Vibration and Accelerometer Data")
    st.markdown("Analyze vibration patterns, critical for detecting **imbalance** or **bearing wear**.")

    vib_col1, vib_col2, vib_col3 = st.columns(3)

    with vib_col1:
        st.markdown(f"""
        <div class="sensor-card">
            <h4>Accel X</h4>
            <h2>{latest['accel_x']:.4f} g</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sensor-card">
            <h4>Accel Y</h4>
            <h2>{latest['accel_y']:.4f} g</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sensor-card">
            <h4>Accel Z</h4>
            <h2>{latest['accel_z']:.4f} g</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with vib_col2:
        st.markdown(f"""
        <div class="sensor-card">
            <h4>Accel Magnitude</h4>
            <h2>{latest['accel_magnitude']:.4f} g</h2>
            <p>Overall vibration intensity.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sensor-card">
            <h4>Vibration RMS</h4>
            <h2>{latest['vibration_rms']:.4f} g</h2>
            <p>Root Mean Square of dynamic vibrations.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sensor-card">
            <h4>Vibration Skew</h4>
            <h2>{latest['vibration_skew']:.3f}</h2>
            <p>Indicates asymmetry in vibration signal.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with vib_col3:
        fig_accel = go.Figure()
        fig_accel.add_trace(go.Scatter(x=df_display['timestamp'], y=df_display['accel_x'], mode='lines', name='Accel X'))
        fig_accel.add_trace(go.Scatter(x=df_display['timestamp'], y=df_display['accel_y'], mode='lines', name='Accel Y'))
        fig_accel.add_trace(go.Scatter(x=df_display['timestamp'], y=df_display['accel_z'], mode='lines', name='Accel Z (incl. Gravity)'))
        fig_accel.update_layout(title='Accelerometer Data (X, Y, Z)', height=300, xaxis_title="Time", yaxis_title="Acceleration (g)")
        st.plotly_chart(fig_accel, use_container_width=True)

        fig_vib_derived = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                       subplot_titles=('Vibration RMS Trend (Dynamic)', 'Vibration Skew Trend'))
        fig_vib_derived.add_trace(go.Scatter(x=df_display['timestamp'], y=df_display['vibration_rms'], mode='lines', name='RMS'), row=1, col=1)
        fig_vib_derived.add_hline(y=simulator.anomaly_thresholds['vibration_rms_threshold'], line_dash="dash", line_color="red", annotation_text="RMS Threshold", row=1, col=1)
        
        fig_vib_derived.add_trace(go.Scatter(x=df_display['timestamp'], y=df_display['vibration_skew'], mode='lines', name='Skew'), row=2, col=1)
        fig_vib_derived.add_hline(y=simulator.anomaly_thresholds['vibration_skew_threshold'], line_dash="dash", line_color="red", annotation_text="Skew Threshold", row=2, col=1)
        fig_vib_derived.add_hline(y=-simulator.anomaly_thresholds['vibration_skew_threshold'], line_dash="dash", line_color="red", annotation_text="-Skew Threshold", row=2, col=1)
        
        fig_vib_derived.update_layout(height=400, showlegend=False)
        fig_vib_derived.update_yaxes(title_text="RMS (g)", row=1, col=1)
        fig_vib_derived.update_yaxes(title_text="Skew", row=2, col=1)
        fig_vib_derived.update_xaxes(title_text="Time", row=2, col=1)
        st.plotly_chart(fig_vib_derived, use_container_width=True)

with tab4: # Rotation Sensors
    st.header("🔄 Rotation and Gyroscope Data")
    st.markdown("Monitor rotational stability and angular velocity for **shaft misalignment** and **RPM stability** issues.")

    gyro_col1, gyro_col2, gyro_col3 = st.columns(3)

    with gyro_col1:
        st.markdown(f"""
        <div class="sensor-card">
            <h4>Gyro X</h4>
            <h2>{latest['gyro_x']:.4f} °/s</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sensor-card">
            <h4>Gyro Y</h4>
            <h2>{latest['gyro_y']:.4f} °/s</h2>
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sensor-card">
            <h4>Gyro Z</h4>
            <h2>{latest['gyro_z']:.4f} °/s</h2>
        </div>
        """, unsafe_allow_html=True)

    with gyro_col2:
        st.markdown(f"""
        <div class="sensor-card">
            <h4>Gyro Magnitude</h4>
            <h2>{latest['gyro_magnitude']:.4f} °/s</h2>
            <p>Overall angular velocity.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sensor-card">
            <h4>Tilt Angle</h4>
            <h2>{latest['tilt_angle']:.2f} °</h2>
            <p>Indicates static or dynamic tilt.</p>
        </div>
        """, unsafe_allow_html=True)

    with gyro_col3:
        fig_gyro = go.Figure()
        fig_gyro.add_trace(go.Scatter(x=df_display['timestamp'], y=df_display['gyro_x'], mode='lines', name='Gyro X'))
        fig_gyro.add_trace(go.Scatter(x=df_display['timestamp'], y=df_display['gyro_y'], mode='lines', name='Gyro Y'))
        fig_gyro.add_trace(go.Scatter(x=df_display['timestamp'], y=df_display['gyro_z'], mode='lines', name='Gyro Z'))
        fig_gyro.update_layout(title='Gyroscope Data (X, Y, Z)', height=300, xaxis_title="Time", yaxis_title="Angular Velocity (°/s)")
        st.plotly_chart(fig_gyro, use_container_width=True)

        fig_gyro_derived = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                         subplot_titles=('Gyroscope Magnitude Trend', 'Tilt Angle Trend'))
        fig_gyro_derived.add_trace(go.Scatter(x=df_display['timestamp'], y=df_display['gyro_magnitude'], mode='lines', name='Gyro Magnitude'), row=1, col=1)
        fig_gyro_derived.add_hline(y=simulator.anomaly_thresholds['gyro_magnitude_threshold'], line_dash="dash", line_color="red", annotation_text="Gyro Magnitude Threshold", row=1, col=1)
        
        fig_gyro_derived.add_trace(go.Scatter(x=df_display['timestamp'], y=df_display['tilt_angle'], mode='lines', name='Tilt Angle'), row=2, col=1)
        fig_gyro_derived.add_hline(y=simulator.anomaly_thresholds['tilt_angle_threshold'], line_dash="dash", line_color="red", annotation_text="Tilt Angle Threshold", row=2, col=1)

        fig_gyro_derived.update_layout(height=400, showlegend=False)
        fig_gyro_derived.update_yaxes(title_text="Magnitude (°/s)", row=1, col=1)
        fig_gyro_derived.update_yaxes(title_text="Angle (°)", row=2, col=1)
        fig_gyro_derived.update_xaxes(title_text="Time", row=2, col=1)
        st.plotly_chart(fig_gyro_derived, use_container_width=True)

with tab5: # ML Analytics
    st.header("🤖 Machine Learning Analytics for Predictive Maintenance")
    st.markdown("Real-time predictions of engine health and anomaly detection using trained ML models.")

    ml_predictions = predictor.predict(df)
    if ml_predictions:
        ml_col1, ml_col2, ml_col3 = st.columns(3)
        with ml_col1:
            ml_health = ml_predictions['health_prediction'].lower()
            ml_card_class = f"{ml_health}-card"
            st.markdown(f"""
            <div class="metric-card {ml_card_class}">
                <h4>ML Health Prediction</h4>
                <h2>{ml_predictions['health_prediction']}</h2>
                <p>Confidence: {ml_predictions['health_confidence']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with ml_col2:
            st.markdown(f"""
            <div class="metric-card {'critical-card' if ml_predictions['anomaly_detected'] else 'normal-card'}">
                <h4>ML Anomaly Detected</h4>
                <h2>{'Yes' if ml_predictions['anomaly_detected'] else 'No'}</h2>
            </div>
            """, unsafe_allow_html=True)
        with ml_col3:
            st.markdown(f"""
            <div class="metric-card normal-card">
                <h4>ML RUL Prediction (hrs)</h4>
                <h2>{ml_predictions['rul_prediction']:,}</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Model Status and Performance")
        if predictor.models_trained:
            st.success("✅ ML Models are currently trained and active.")
            st.info(f"Last training data points: {len(df)}")
            
            # Check if full_health_probabilities is available
            if 'full_health_probabilities' in ml_predictions and predictor.svm_model and hasattr(predictor.svm_model, 'predict_proba'):
                try:
                    # Use the full_health_probabilities array returned by the predict method
                    # And use the predictor's own health_labels for x-axis
                    fig_pred_health = px.bar(
                        x=predictor.health_labels,
                        y=ml_predictions['full_health_probabilities'] * 100,
                        labels={'x': 'Health Status', 'y': 'Confidence (%)'},
                        title='ML Model Health Prediction Confidence',
                        color=predictor.health_labels,
                        color_discrete_map={
                            'Normal': '#27ae60',
                            'Caution': '#f39c12',
                            'Warning': '#e67e22',
                            'Critical': '#e74c3c'
                        }
                    )
                    fig_pred_health.update_layout(yaxis_range=[0,100])
                    st.plotly_chart(fig_pred_health, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate health prediction confidence chart: {e}. Check if data has enough variation for all health categories.")
            else:
                st.warning("SVM model not available or has not produced comprehensive probability data for confidence visualization.")

        else:
            st.warning("⚠️ ML Models are not yet trained or require more data. Click 'Train ML Models' in the sidebar.")
            st.info(f"Minimum samples required for training: {MIN_TRAINING_SAMPLES}. Current samples: {len(df)}")
    else:
        st.info("No ML predictions available. Ensure models are trained and data is flowing.")

with tab6: # Historical Data
    st.header("📊 Full Historical Sensor Data")
    st.markdown("Review the raw historical data collected from the engine sensors.")

    if show_raw_data:
        st.subheader("Raw Sensor Data Table")
        st.dataframe(df)
    else:
        st.info("Toggle 'Show Raw Data' in the sidebar to view the full dataset.")
    
    st.subheader("Data Export")
    if not df.empty:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name="engine_sensor_data.csv",
            mime="text/csv",
        )
    else:
        st.warning("No data to download yet.")

with tab7: # Sensor Thresholds
    st.header("⚙️ Sensor Anomaly Threshold Configuration")
    st.markdown("Adjust the thresholds used by the simulator to calculate the anomaly score and determine health status.")
    st.markdown("Changes here will immediately affect the `Anomaly Score` calculation and `Health Status`.")

    st.subheader("Anomaly Scoring Thresholds")
    
    col_thresh1, col_thresh2 = st.columns(2)

    with col_thresh1:
        st.session_state.anomaly_thresholds['vibration_rms_threshold'] = st.slider(
            "Vibration RMS Anomaly Threshold (g)",
            min_value=0.01, max_value=1.0,
            value=st.session_state.anomaly_thresholds['vibration_rms_threshold'], step=0.01
        )
        st.session_state.anomaly_thresholds['temp_threshold'] = st.slider(
            "Engine Temperature Anomaly Threshold (°C)",
            min_value=90, max_value=150,
            value=st.session_state.anomaly_thresholds['temp_threshold'], step=1
        )
        st.session_state.anomaly_thresholds['rpm_deviation_threshold'] = st.slider(
            "RPM Deviation Anomaly Threshold (RPM)",
            min_value=50, max_value=500,
            value=st.session_state.anomaly_thresholds['rpm_deviation_threshold'], step=10
        )
        st.session_state.anomaly_thresholds['gyro_magnitude_threshold'] = st.slider(
            "Gyroscope Magnitude Anomaly Threshold (°/s)",
            min_value=0.5, max_value=15.0,
            value=st.session_state.anomaly_thresholds['gyro_magnitude_threshold'], step=0.1
        )

    with col_thresh2:
        st.session_state.anomaly_thresholds['tilt_angle_threshold'] = st.slider(
            "Tilt Angle Anomaly Threshold (°)",
            min_value=5, max_value=45,
            value=st.session_state.anomaly_thresholds['tilt_angle_threshold'], step=1
        )
        st.session_state.anomaly_thresholds['vibration_skew_threshold'] = st.slider(
            "Vibration Skewness Anomaly Threshold (abs)",
            min_value=0.1, max_value=5.0,
            value=st.session_state.anomaly_thresholds['vibration_skew_threshold'], step=0.05
        )

        st.markdown("---")
        st.subheader("Health Status Scoring Thresholds")
        st.session_state.anomaly_thresholds['caution_score'] = st.slider(
            "Caution Anomaly Score Threshold",
            min_value=0, max_value=100,
            value=st.session_state.anomaly_thresholds['caution_score'], step=1
        )
        st.session_state.anomaly_thresholds['warning_score'] = st.slider(
            "Warning Anomaly Score Threshold",
            min_value=0, max_value=100,
            value=st.session_state.anomaly_thresholds['warning_score'], step=1
        )
        st.session_state.anomaly_thresholds['critical_score'] = st.slider(
            "Critical Anomaly Score Threshold",
            min_value=0, max_value=100,
            value=st.session_state.anomaly_thresholds['critical_score'], step=1
        )

    st.markdown("---")
    st.subheader("Remaining Useful Life (RUL) Configuration")
    st.session_state.anomaly_thresholds['base_rul'] = st.slider(
        "Base RUL (hours, for healthy engine)",
        min_value=1000, max_value=20000,
        value=st.session_state.anomaly_thresholds['base_rul'], step=100
    )
    st.session_state.anomaly_thresholds['rul_impact_factor'] = st.slider(
        "RUL Impact Factor (Anomaly Score Multiplier)",
        min_value=1, max_value=200,
        value=st.session_state.anomaly_thresholds['rul_impact_factor'], step=1
    )

    if st.button("Apply Sensor Threshold Changes", key="apply_thresholds_button"):
        simulator.update_anomaly_thresholds(st.session_state.anomaly_thresholds)
        st.success("Sensor anomaly thresholds updated! Rerunning simulation with new thresholds.")
        st.rerun()

# Auto-refresh mechanism
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()