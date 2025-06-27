import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
import logging

# Import for auto-refresh functionality
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'csv_file': 'brake_system_data.csv',
    'ml_results_file': 'ml_predictions.json', # File to store ML results
    'sim_interval': 10,   # Faster updates for realistic simulation
    'model_retrain_interval': 20, # How often ML model retrains (seconds)
    'data_window': 100 # Number of data points to display in historical charts
}

# Define consistent thresholds for visual indicators and warnings
THRESHOLDS = {
    'pressure': {'normal_min': 70, 'warning_min': 50, 'critical_min': 45, 'gauge_max': 120, 'gauge_ref': 85},
    'temperature': {'normal_max': 80, 'warning_max': 100, 'critical_max': 120, 'gauge_max': 150, 'gauge_ref': 55},
    'pad_wear': {'normal_max': 60, 'warning_max': 75, 'critical_max': 90, 'gauge_max': 100, 'gauge_ref': 25},
    'response_time': {'normal_max': 0.5, 'warning_max': 0.7, 'critical_max': 0.8, 'gauge_max': 2.0, 'gauge_ref': 0.35}
}

# Sensor configuration for dashboard organization
SENSOR_CONFIG = {
    'brake_pressure': {
        'name': 'Brake Pressure',
        'unit': 'PSI',
        'icon': '🔧',
        'threshold_key': 'pressure',
        'description': 'Hydraulic pressure in brake system',
        'normal_range': '70-120 PSI'
    },
    'brake_temperature': {
        'name': 'Brake Temperature',
        'unit': '°C',
        'icon': '🌡️',
        'threshold_key': 'temperature',
        'description': 'Temperature of brake components',
        'normal_range': '25-80°C'
    },
    'brake_pad_wear': {
        'name': 'Brake Pad Wear',
        'unit': '%',
        'icon': '🔩',
        'threshold_key': 'pad_wear',
        'description': 'Percentage wear of brake pads',
        'normal_range': '0-60%'
    },
    'brake_response_time': {
        'name': 'Response Time',
        'unit': 'seconds',
        'icon': '⏱️',
        'threshold_key': 'response_time',
        'description': 'Time for brake system to respond',
        'normal_range': '0.1-0.5s'
    }
}


def initialize_session_state():
    """Initializes all necessary session state variables for Streamlit."""
    # ML Prediction state
    if 'latest_prediction' not in st.session_state:
        st.session_state.latest_prediction = {'condition': 'No Prediction', 'confidence': 0.0, 'rul': 0}
    if 'model_accuracy' not in st.session_state:
        st.session_state.model_accuracy = 0
    if 'anomaly_count' not in st.session_state: # Not currently used in the new simulation logic, but kept for consistency
        st.session_state.anomaly_count = 0
    if 'manual_override' not in st.session_state: # Not currently used in the new simulation logic, but kept for consistency
        st.session_state.manual_override = {'enabled': False, 'condition': 'Normal'}

    # Fault simulation state
    if 'fault_state' not in st.session_state:
        st.session_state.fault_state = {
            'fluid_leak': {'active': False, 'progress': 0, 'start_time': None},
            'overheating': {'active': False, 'progress': 0, 'start_time': None},
            'pad_wear': {'active': False, 'progress': THRESHOLDS['pad_wear']['gauge_ref'], 'start_time': None}, # Initial normal wear
            'delayed_response': {'active': False, 'progress': 0, 'start_time': None}
        }
    
    # Base sensor values for normal operation
    if 'base_values' not in st.session_state:
        st.session_state.base_values = {
            'brake_pressure': 85,
            'brake_temperature': 55,
            'brake_pad_wear': 25,
            'brake_response_time': 0.35
        }

    # System control flags
    if 'emergency_brake_active' not in st.session_state:
        st.session_state.emergency_brake_active = False
    if 'system_mode' not in st.session_state:
        st.session_state.system_mode = 'Normal'
    if 'fault_injection_enabled' not in st.session_state:
        st.session_state.fault_injection_enabled = True
    
    # Overall system health (derived from sensor data and active faults)
    if 'current_system_health' not in st.session_state:
        st.session_state.current_system_health = 'Normal'

    # Threading control for ML training
    if 'ml_thread_running' not in st.session_state:
        st.session_state.ml_thread_running = False
    
    # Placeholder for last ML train time (not strictly critical for operation, but useful for display)
    if 'last_ml_train_time' not in st.session_state:
        st.session_state.last_ml_train_time = datetime.now()


def determine_system_health(data, current_session_state):
    """
    Determines the overall system health based on sensor data and active faults.
    `current_session_state` refers to st.session_state directly.
    """
    active_faults = [name for name, fault in current_session_state['fault_state'].items() if fault['active']]
    
    if active_faults:
        max_severity = 0
        for fault_name in active_faults:
            fault = current_session_state['fault_state'][fault_name]
            progress = fault['progress']
            
            if fault_name == 'fluid_leak':
                severity = min(progress / 100, 1.0)
            elif fault_name == 'overheating':
                severity = min(progress / 100, 1.0)
            elif fault_name == 'pad_wear':
                if progress > THRESHOLDS['pad_wear']['critical_max']:
                    severity = 1.0
                elif progress > THRESHOLDS['pad_wear']['warning_max']:
                    severity = 0.6
                else:
                    severity = 0.3
            elif fault_name == 'delayed_response':
                severity = min(progress / 100, 1.0)
            else:
                severity = 0.5 # Default severity if not specified
            
            max_severity = max(max_severity, severity)
        
        if max_severity > 0.7:
            return 'Fault'
        elif max_severity > 0.3:
            return 'Warning'
        else:
            return 'Warning' if max_severity > 0 else 'Normal' # If fault is active but low progress, still warning
    
    # If no faults are explicitly active, check sensor readings against thresholds
    if (data['brake_pressure'] < THRESHOLDS['pressure']['critical_min'] or
        data['brake_temperature'] > THRESHOLDS['temperature']['critical_max'] or
        data['brake_pad_wear'] > THRESHOLDS['pad_wear']['critical_max'] or
        data['brake_response_time'] > THRESHOLDS['response_time']['critical_max']):
        return 'Fault'
    
    if (data['brake_pressure'] < THRESHOLDS['pressure']['warning_min'] or
        data['brake_temperature'] > THRESHOLDS['temperature']['warning_max'] or
        data['brake_pad_wear'] > THRESHOLDS['pad_wear']['warning_max'] or
        data['brake_response_time'] > THRESHOLDS['response_time']['warning_max']):
        return 'Warning'
    
    return 'Normal'


def generate_brake_data(session_state_context, current_timestep):
    """
    Generates realistic brake sensor data with optional fault injection.
    Accesses st.session_state_context directly, as it's called from the main loop.
    """
    # Base normal values with realistic noise, directly from session_state
    data = {
        'brake_pressure': np.random.normal(session_state_context['base_values']['brake_pressure'], 2),
        'brake_temperature': np.random.normal(session_state_context['base_values']['brake_temperature'], 1.5),
        'brake_pad_wear': session_state_context['base_values']['brake_pad_wear'] + np.random.normal(0, 0.5),
        'brake_response_time': np.random.normal(session_state_context['base_values']['brake_response_time'], 0.02)
    }

    # Apply emergency braking effects if activated by the user
    if session_state_context['emergency_brake_active']:
        data['brake_pressure'] += 15 + np.random.normal(0, 3)
        data['brake_temperature'] += 5 + np.random.normal(0, 2)
        data['brake_response_time'] *= 0.8
        data['brake_response_time'] = max(0.1, data['brake_response_time'])

    # Default condition for this data point
    condition = "Normal" 

    # Apply fault effects ONLY if fault injection is enabled
    if session_state_context['fault_injection_enabled']:
        for fault_type, fault_info in session_state_context['fault_state'].items():
            if fault_info['active']:
                # Update progress for gradual faults
                # Use a simple time progression for demo purposes
                if fault_info['start_time'] is not None:
                    time_elapsed = time.time() - fault_info['start_time']
                    if fault_type == 'fluid_leak':
                        # Progress from 10 to 100 over approx 100 units of time
                        fault_info['progress'] = min(100.0, fault_info['progress'] + np.random.uniform(0.5, 2.0))
                    elif fault_type == 'overheating':
                        fault_info['progress'] = min(100.0, fault_info['progress'] + np.random.uniform(1.0, 3.0))
                    elif fault_type == 'pad_wear':
                        # Pad wear has a base value, progress increments from initial value
                        fault_info['progress'] = min(100.0, fault_info['progress'] + np.random.uniform(0.1, 0.5))
                    elif fault_type == 'delayed_response':
                        fault_info['progress'] = min(100.0, fault_info['progress'] + np.random.uniform(0.8, 2.5))
                
                # Apply fault-specific data modifications
                if fault_type == 'fluid_leak':
                    leak_severity = min(fault_info['progress'] / 100, 1.0)
                    data['brake_pressure'] -= 50 * leak_severity + np.random.normal(0, 5)
                    data['brake_temperature'] += 5 * leak_severity + np.random.normal(0, 1)
                    data['brake_response_time'] += 0.3 * leak_severity + np.random.normal(0, 0.05)

                elif fault_type == 'overheating':
                    heat_severity = min(fault_info['progress'] / 100, 1.0)
                    data['brake_temperature'] += 60 * heat_severity + np.random.normal(0, 5)
                    pressure_variation = 10 * heat_severity * np.sin(current_timestep * 0.1) # Using current_timestep for dynamic effect
                    data['brake_pressure'] += pressure_variation + np.random.normal(0, 2)
                    if data['brake_temperature'] > 90:
                        data['brake_response_time'] += 0.1 * heat_severity + np.random.normal(0, 0.03)

                elif fault_type == 'pad_wear':
                    wear_level = fault_info['progress'] # This is the actual pad wear percentage
                    data['brake_pad_wear'] = wear_level + np.random.normal(0, 1)
                    if wear_level > 80:
                        wear_factor = (wear_level - 80) / 20
                        data['brake_response_time'] += 0.4 * wear_factor + np.random.normal(0, 0.05)
                        data['brake_pressure'] += 8 * wear_factor + np.random.normal(0, 3)
                        data['brake_temperature'] += 15 * wear_factor + np.random.normal(0, 2)

                elif fault_type == 'delayed_response':
                    delay_severity = min(fault_info['progress'] / 100, 1.0)
                    response_delay = 0.8 * delay_severity + np.random.normal(0, 0.08)
                    data['brake_response_time'] += response_delay
                    if delay_severity > 0.5:
                        data['brake_pressure'] -= 10 * (delay_severity - 0.5) + np.random.normal(0, 2)

    # Ensure realistic bounds for all sensors after applying all effects
    data['brake_pressure'] = max(20, min(120, data['brake_pressure']))
    data['brake_temperature'] = max(25, min(150, data['brake_temperature']))
    data['brake_pad_wear'] = max(0, min(100, data['brake_pad_wear']))
    data['brake_response_time'] = max(0.1, min(2.0, data['brake_response_time']))

    # Update system health based on current data and active faults from session state
    session_state_context['current_system_health'] = determine_system_health(data, session_state_context)
    data['condition'] = session_state_context['current_system_health'] # Reflect overall health in the data record

    return data


def inject_fault(fault_type, mode='gradual', session_state_context=None):
    """Injects a specific fault into the system's state."""
    if session_state_context is None:
        session_state_context = st.session_state # Use st.session_state directly here as this is a Streamlit callback

    if not session_state_context['fault_injection_enabled']:
        st.warning("Cannot inject fault: Fault Injection is currently disabled.")
        return

    current_time = time.time()

    # Reset other faults
    for fault_key in session_state_context['fault_state']:
        if fault_key != fault_type:
            session_state_context['fault_state'][fault_key]['active'] = False
            session_state_context['fault_state'][fault_key]['progress'] = 0
            session_state_context['fault_state'][fault_key]['start_time'] = None

    # Activate the selected fault
    if fault_type == 'no_fault':
        reset_system(session_state_context) # Use the comprehensive reset function
        st.success("System set to No Faults (Normal Conditions)!")
        session_state_context['system_mode'] = 'Normal'
        session_state_context['current_system_health'] = 'Normal'
        return
    
    target_fault = session_state_context['fault_state'].get(fault_type)
    if target_fault:
        target_fault['active'] = True
        target_fault['start_time'] = current_time
        
        if fault_type == 'fluid_leak':
            target_fault['progress'] = 60 if mode == 'sudden' else 10
        elif fault_type == 'overheating':
            target_fault['progress'] = 70 if mode == 'sudden' else 5
        elif fault_type == 'pad_wear':
            target_fault['progress'] = 92 if mode == 'sudden' else THRESHOLDS['pad_wear']['warning_max'] + 2 # Start in warning
        elif fault_type == 'delayed_response':
            target_fault['progress'] = 80 if mode == 'sudden' else 20
        
        st.success(f"Applied {fault_type.replace('_', ' ').title()} ({mode})!")
        session_state_context['current_system_health'] = 'Warning' # Assume any fault injection implies at least warning
        session_state_context['system_mode'] = 'Testing'
    else:
        st.error(f"Unknown fault type: {fault_type}")


def reset_system(session_state_context=None):
    """Resets all faults and emergency brake status to normal operation."""
    if session_state_context is None:
        session_state_context = st.session_state

    # Reset all fault states
    for fault in session_state_context['fault_state'].values():
        fault['active'] = False
        fault['progress'] = 0
        fault['start_time'] = None

    # Reset specific initial value for pad wear
    session_state_context['fault_state']['pad_wear']['progress'] = THRESHOLDS['pad_wear']['gauge_ref']
    session_state_context['emergency_brake_active'] = False
    session_state_context['system_mode'] = 'Normal'
    session_state_context['current_system_health'] = 'Normal'

    # Clear the CSV file on a full reset to ensure fresh data
    if os.path.exists(CONFIG['csv_file']):
        os.remove(CONFIG['csv_file'])
        logger.info("CSV data removed.")
    # Always ensure the CSV is created with headers for a clean start
    pd.DataFrame(columns=[
        'timestamp', 'brake_pressure', 'brake_temperature',
        'brake_pad_wear', 'brake_response_time', 'condition'
    ]).to_csv(CONFIG['csv_file'], index=False)
    logger.info("CSV data file recreated with headers.")

    # Clear ML results file as well
    if os.path.exists(CONFIG['ml_results_file']):
        os.remove(CONFIG['ml_results_file'])
        logger.info("ML results file removed.")
    # Reset ML predictions in session state
    st.session_state.latest_prediction = {'condition': 'No Prediction', 'confidence': 0.0, 'rul': 0}
    st.session_state.model_accuracy = 0
    logger.info("ML results cleared from session state.")


def clean_and_validate_data(df):
    """Cleans and validates the brake system data loaded from CSV."""
    try:
        # Filter rows where timestamp might be too short (e.g., malformed)
        df = df[df['timestamp'].astype(str).str.len() >= 10].copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        numeric_cols = ['brake_pressure', 'brake_temperature', 'brake_pad_wear', 'brake_response_time']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna(subset=numeric_cols)
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    except Exception as e:
        logger.error(f"Error in data cleaning: {str(e)}")
        # Return an empty DataFrame with expected columns to prevent further errors
        return pd.DataFrame(columns=[
            'timestamp', 'brake_pressure', 'brake_temperature',
            'brake_pad_wear', 'brake_response_time', 'condition'
        ])


def save_ml_results(prediction, accuracy):
    """Saves ML prediction and accuracy to a JSON file."""
    results = {
        'latest_prediction': prediction,
        'model_accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    try:
        with open(CONFIG['ml_results_file'], 'w') as f:
            json.dump(results, f)
        logger.debug("ML results saved to file.")
    except Exception as e:
        logger.error(f"Error saving ML results to file: {e}")

def load_ml_results():
    """Loads ML prediction and accuracy from a JSON file."""
    if os.path.exists(CONFIG['ml_results_file']):
        try:
            with open(CONFIG['ml_results_file'], 'r') as f:
                results = json.load(f)
            logger.debug("ML results loaded from file.")
            return results['latest_prediction'], results['model_accuracy']
        except Exception as e:
            logger.error(f"Error loading ML results from file: {e}")
            return None, None
    return None, None


def ml_training_thread_func():
    """
    Runs ML training in a separate thread and saves results to a file.
    This function *must not* directly access st.session_state.
    """
    last_train_time = time.time()
    while True: # This thread runs indefinitely as a daemon
        try:
            # Only train if enough time has passed since last training or if file does not exist
            if time.time() - last_train_time < CONFIG['model_retrain_interval']:
                time.sleep(1) # Sleep briefly if not yet time to retrain
                continue

            if not os.path.exists(CONFIG['csv_file']):
                logger.info("ML Training: Data CSV not found. Waiting for data.")
                time.sleep(5)
                continue

            df = pd.read_csv(CONFIG['csv_file'])
            df = clean_and_validate_data(df)

            if len(df) < 50: # Minimum samples for training
                logger.info(f"ML Training: Insufficient data for training: {len(df)} rows. Waiting...")
                time.sleep(CONFIG['model_retrain_interval'])
                continue

            feature_cols = ['brake_pressure', 'brake_temperature', 'brake_pad_wear', 'brake_response_time']
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0 # Add missing column with default value to prevent errors

            X = df[feature_cols].values
            y = df['condition'].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            unique_classes = np.unique(y)
            if len(X_scaled) < 2 or len(unique_classes) < 2:
                logger.warning("ML Training: Not enough unique classes or samples for meaningful training. Skipping this cycle.")
                save_ml_results({'condition': 'No Prediction', 'confidence': 0.0, 'rul': 0}, 0.0)
                last_train_time = time.time() # Update last train time even if skipped
                time.sleep(CONFIG['model_retrain_interval'])
                continue

            if len(unique_classes) > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                # If only one class, cannot stratify
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )

            models = {
                'svm': SVC(probability=True, random_state=42),
                'rf': RandomForestClassifier(n_estimators=50, random_state=42)
            }

            best_model = None
            best_accuracy = 0

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

            latest_prediction_data = {'condition': 'Unknown', 'confidence': 0.0, 'rul': 0}
            if best_model and len(X_scaled) > 0:
                latest_features = X_scaled[-1].reshape(1, -1)
                prediction_class = best_model.predict(latest_features)[0]
                confidence = max(best_model.predict_proba(latest_features)[0])

                # Dynamic RUL calculation based on latest sensor data
                latest_data_row = df.iloc[-1]
                rul_map = {'Normal': 800, 'Warning': 200, 'Fault': 50}
                base_rul = rul_map.get(prediction_class, 300)
                
                estimated_rul = base_rul
                if latest_data_row['brake_pad_wear'] > THRESHOLDS['pad_wear']['normal_max']:
                    wear_factor = (latest_data_row['brake_pad_wear'] - THRESHOLDS['pad_wear']['normal_max']) / (THRESHOLDS['pad_wear']['critical_max'] - THRESHOLDS['pad_wear']['normal_max'])
                    estimated_rul = min(estimated_rul, base_rul * (1 - min(1, wear_factor * 0.8)))
                
                if latest_data_row['brake_pressure'] < THRESHOLDS['pressure']['normal_min']:
                    pressure_factor = (THRESHOLDS['pressure']['normal_min'] - latest_data_row['brake_pressure']) / (THRESHOLDS['pressure']['normal_min'] - THRESHOLDS['pressure']['critical_min'])
                    estimated_rul = min(estimated_rul, base_rul * (1 - min(1, pressure_factor * 0.8)))

                if latest_data_row['brake_temperature'] > THRESHOLDS['temperature']['normal_max']:
                    temp_factor = (latest_data_row['brake_temperature'] - THRESHOLDS['temperature']['normal_max']) / (THRESHOLDS['temperature']['critical_max'] - THRESHOLDS['temperature']['normal_max'])
                    estimated_rul = min(estimated_rul, base_rul * (1 - min(1, temp_factor * 0.8)))

                if latest_data_row['brake_response_time'] > THRESHOLDS['response_time']['normal_max']:
                    response_factor = (latest_data_row['brake_response_time'] - THRESHOLDS['response_time']['normal_max']) / (THRESHOLDS['response_time']['critical_max'] - THRESHOLDS['response_time']['normal_max'])
                    estimated_rul = min(estimated_rul, base_rul * (1 - min(1, response_factor * 0.8)))

                latest_prediction_data = {
                    'condition': prediction_class,
                    'confidence': round(confidence * 100, 1),
                    'rul': max(0, int(estimated_rul + np.random.randint(-30, 30)))
                }
                logger.info(f"ML trained. Prediction: {latest_prediction_data['condition']}, RUL: {latest_prediction_data['rul']}")
            
            save_ml_results(latest_prediction_data, round(best_accuracy * 100, 1))
            last_train_time = time.time() # Update last train time

        except Exception as e:
            logger.error(f"ML Training Thread Error: {str(e)}")
            save_ml_results({'condition': 'Error', 'confidence': 0.0, 'rul': 0}, 0.0) # Save error state
            last_train_time = time.time() # Still update time to avoid constant errors

        time.sleep(CONFIG['model_retrain_interval'])

    logger.info("ML training thread stopped.")


def create_sensor_dashboard_section(sensor_key, sensor_config, latest_data, df):
    """Creates a comprehensive dashboard section for a single sensor."""
    
    sensor_name = sensor_config['name']
    unit = sensor_config['unit']
    icon = sensor_config['icon']
    threshold_key = sensor_config['threshold_key']
    description = sensor_config['description']
    normal_range = sensor_config['normal_range']
    
    current_value = latest_data[sensor_key]
    thresholds = THRESHOLDS[threshold_key]
    
    status_text = ""
    status_color_code = "" # For Streamlit markdown color
    gauge_bar_color = "lightgreen" # Default gauge bar color

    # Dynamically determine status and gauge color based on thresholds
    # This logic needs to be correctly applied for pressure (lower is critical) and others (higher is critical)
    gauge_steps = []
    gauge_threshold_value = 0

    if threshold_key == 'pressure':  # Lower value is worse (red at low end)
        if current_value < thresholds['critical_min']:
            status_text = "Critical"
            status_color_code = "red"
            gauge_bar_color = "red"
        elif current_value < thresholds['warning_min']:
            status_text = "Warning"
            status_color_code = "orange"
            gauge_bar_color = "orange"
        else:
            status_text = "Normal"
            status_color_code = "green"
            gauge_bar_color = "green"
        
        gauge_steps = [
            {'range': [0, thresholds['critical_min']], 'color': "#FF0000"}, # Red
            {'range': [thresholds['critical_min'], thresholds['warning_min']], 'color': "#FFA500"}, # Orange
            {'range': [thresholds['warning_min'], thresholds['gauge_max']], 'color': "#90EE90"} # Light Green
        ]
        gauge_threshold_value = thresholds['critical_min']

    else:  # Higher value is worse (red at high end)
        if current_value > thresholds['critical_max']:
            status_text = "Critical"
            status_color_code = "red"
            gauge_bar_color = "red"
        elif current_value > thresholds['warning_max']:
            status_text = "Warning"
            status_color_code = "orange"
            gauge_bar_color = "orange"
        else:
            status_text = "Normal"
            status_color_code = "green"
            gauge_bar_color = "green"
        
        gauge_steps = [
            {'range': [0, thresholds['normal_max']], 'color': "#90EE90"}, # Light Green
            {'range': [thresholds['normal_max'], thresholds['warning_max']], 'color': "#FFA500"}, # Orange
            {'range': [thresholds['warning_max'], thresholds['gauge_max']], 'color': "#FF0000"} # Red
        ]
        gauge_threshold_value = thresholds['critical_max']
    
    st.markdown(f"### {icon} {sensor_name}")
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        st.metric(
            label=f"Current {sensor_name}",
            value=f"{current_value:.2f} {unit}",
            delta=None
        )
        st.markdown(f"**Status:** :{status_color_code}[{status_text}]")
        st.markdown(f"**Normal Range:** {normal_range}")
        st.markdown(f"*{description}*")
    
    with col2:
        # Corrected: 'gauge' properties are directly in go.Indicator
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_value,
            title={'text': f"{sensor_name} ({unit})"},
            gauge={
                'axis': {'range': [0, thresholds['gauge_max']], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': gauge_bar_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': gauge_steps, # Use dynamically set steps
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': gauge_threshold_value # Use dynamically set threshold value
                }
            }
        ))
        
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{sensor_key}")
    
    with col3:
        display_df = df.tail(CONFIG['data_window'])
        
        fig_trend = px.line(
            display_df, 
            x='timestamp', 
            y=sensor_key,
            title=f"{sensor_name} Trend (Last {len(display_df)} readings)",
            labels={sensor_key: f"{sensor_name} ({unit})", 'timestamp': 'Time'}
        )
        
        # Add threshold lines to the trend chart
        if threshold_key == 'pressure':
            fig_trend.add_hline(y=thresholds['normal_min'], line_dash="dot", line_color="green", 
                                 annotation_text="Normal Min", annotation_position="top right")
            fig_trend.add_hline(y=thresholds['warning_min'], line_dash="dash", line_color="orange", 
                                 annotation_text="Warning Min", annotation_position="top left")
            fig_trend.add_hline(y=thresholds['critical_min'], line_dash="dash", line_color="red", 
                                 annotation_text="Critical Min", annotation_position="bottom right")
        else:
            fig_trend.add_hline(y=thresholds['normal_max'], line_dash="dot", line_color="green", 
                                 annotation_text="Normal Max", annotation_position="top right")
            fig_trend.add_hline(y=thresholds['warning_max'], line_dash="dash", line_color="orange", 
                                 annotation_text="Warning Max", annotation_position="top left")
            fig_trend.add_hline(y=thresholds['critical_max'], line_dash="dash", line_color="red", 
                                 annotation_text="Critical Max", annotation_position="bottom right")
        
        fig_trend.update_traces(line_color=gauge_bar_color, line_width=3)
        fig_trend.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_{sensor_key}")
    
    with st.expander(f"📊 {sensor_name} Detailed Analysis", expanded=False):
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            recent_data = display_df[sensor_key].tail(20)
            st.markdown("**Recent Statistics (Last 20 readings):**")
            st.write(f"- Average: {recent_data.mean():.2f} {unit}")
            st.write(f"- Min: {recent_data.min():.2f} {unit}")
            st.write(f"- Max: {recent_data.max():.2f} {unit}")
        with col_stats2:
            st.markdown("**Detailed Thresholds:**")
            if threshold_key == 'pressure':
                st.write(f"- Normal: $\\geq{thresholds['normal_min']}$ {unit}")
                st.write(f"- Warning: $\\geq{thresholds['warning_min']}$ and $<{thresholds['normal_min']}$ {unit}")
                st.write(f"- Critical: $<{thresholds['critical_min']}$ {unit}")
            else:
                st.write(f"- Normal: $\\leq{thresholds['normal_max']}$ {unit}")
                st.write(f"- Warning: $>{thresholds['normal_max']}$ and $\\leq{thresholds['warning_max']}$ {unit}")
                st.write(f"- Critical: $>{thresholds['critical_max']}$ {unit}")
    st.markdown("---")


def generate_and_save_data():
    """Generate new brake data point and save to CSV"""
    # Initialize timestep if not exists
    if 'simulation_timestep' not in st.session_state:
        st.session_state.simulation_timestep = 0
    
    current_time = datetime.now()
    
    # Increment timestep for dynamic effects
    st.session_state.simulation_timestep += 1

    # Generate new data point
    brake_data = generate_brake_data(st.session_state, current_timestep=st.session_state.simulation_timestep)
    
    # Create DataFrame for current reading
    new_row = pd.DataFrame([[
        current_time.strftime('%Y-%m-%d %H:%M:%S'),
        round(brake_data['brake_pressure'], 2),
        round(brake_data['brake_temperature'], 2),
        round(brake_data['brake_pad_wear'], 1),
        round(brake_data['brake_response_time'], 3),
        brake_data['condition']
    ]], columns=[
        'timestamp', 'brake_pressure', 'brake_temperature',
        'brake_pad_wear', 'brake_response_time', 'condition'
    ])

    # Append to CSV
    try:
        new_row.to_csv(CONFIG['csv_file'], mode='a', header=False, index=False)
    except Exception as e:
        logger.error(f"Error appending data to CSV: {e}")


def render_dashboard_content(show_raw_data, show_ml_metrics):
    """Render all dashboard content tabs"""
    # Load all data for display and ML
    df = pd.DataFrame()
    try:
        if os.path.exists(CONFIG['csv_file']):
            df = pd.read_csv(CONFIG['csv_file'])
            df = clean_and_validate_data(df)
        
        if df.empty:
            st.info("No data yet. Please wait for the simulation to start. (Data accumulating...)")
            return
        
        latest_data = df.iloc[-1]

    except pd.errors.EmptyDataError:
        st.info("CSV file is empty. Waiting for simulation data to be written...")
        return
    except Exception as e:
        st.error(f"Error loading or processing data for display: {e}. Please try resetting the system using the sidebar button.")
        return

    # Load ML results from file (updated by the background ML thread)
    pred_from_file, accuracy_from_file = load_ml_results()
    if pred_from_file is not None and accuracy_from_file is not None:
        st.session_state.latest_prediction = pred_from_file
        st.session_state.model_accuracy = accuracy_from_file

    # Create dashboard tabs
    tab_overview, tab_pressure, tab_temp, tab_pad_wear, tab_response_time, tab_predictive = st.tabs([
        "Overview", 
        f"{SENSOR_CONFIG['brake_pressure']['icon']} {SENSOR_CONFIG['brake_pressure']['name']}",
        f"{SENSOR_CONFIG['brake_temperature']['icon']} {SENSOR_CONFIG['brake_temperature']['name']}",
        f"{SENSOR_CONFIG['brake_pad_wear']['icon']} {SENSOR_CONFIG['brake_pad_wear']['name']}",
        f"{SENSOR_CONFIG['brake_response_time']['icon']} {SENSOR_CONFIG['brake_response_time']['name']}",
        "🧠 Predictive Analytics"
    ])

    with tab_overview:
        st.header("✨ Overall System Status & Key Metrics")
        overall_health_color = "green"
        if st.session_state.current_system_health == 'Warning':
            overall_health_color = "orange"
        elif st.session_state.current_system_health == 'Fault':
            overall_health_color = "red"
        
        st.markdown(f"""
        <div style="background-color: {overall_health_color}; padding: 15px; border-radius: 10px; text-align: center; color: white;">
            <h3>Overall Brake System Health: <b>{st.session_state.current_system_health}</b></h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        st.subheader("Current Key Sensor Readings")
        cols_metrics = st.columns(4)
        
        for i, (key, s_config) in enumerate(SENSOR_CONFIG.items()):
            current_val = latest_data[key]
            cols_metrics[i].metric(
                label=f"{s_config['icon']} {s_config['name']}",
                value=f"{current_val:.2f} {s_config['unit']}"
            )
        st.markdown("---")

        st.subheader("Recent Data Trends (All Sensors)")
        fig_all_trends = make_subplots(rows=len(SENSOR_CONFIG), cols=1, 
                                     shared_xaxes=True, 
                                     vertical_spacing=0.08,
                                     subplot_titles=[f"{s['name']} ({s['unit']})" for s in SENSOR_CONFIG.values()])
        
        display_df_all = df.tail(CONFIG['data_window'])

        for i, (key, s_config) in enumerate(SENSOR_CONFIG.items()):
            current_sensor_value = latest_data[key]
            
            line_color = "green"
            if s_config['threshold_key'] == 'pressure':
                if current_sensor_value < THRESHOLDS[s_config['threshold_key']]['critical_min']:
                    line_color = "red"
                elif current_sensor_value < THRESHOLDS[s_config['threshold_key']]['warning_min']:
                    line_color = "orange"
            else:
                if current_sensor_value > THRESHOLDS[s_config['threshold_key']]['critical_max']:
                    line_color = "red"
                elif current_sensor_value > THRESHOLDS[s_config['threshold_key']]['warning_max']:
                    line_color = "orange"

            fig_all_trends.add_trace(go.Scatter(
                x=display_df_all['timestamp'], 
                y=display_df_all[key], 
                name=s_config['name'], 
                mode='lines', 
                line=dict(color=line_color, width=2),
                showlegend=True
            ), row=i+1, col=1)

            if s_config['threshold_key'] == 'pressure':
                fig_all_trends.add_hline(y=THRESHOLDS[s_config['threshold_key']]['normal_min'], line_dash="dot", line_color="lightgreen", row=i+1, col=1)
                fig_all_trends.add_hline(y=THRESHOLDS[s_config['threshold_key']]['warning_min'], line_dash="dash", line_color="orange", row=i+1, col=1)
                fig_all_trends.add_hline(y=THRESHOLDS[s_config['threshold_key']]['critical_min'], line_dash="dash", line_color="red", row=i+1, col=1)
            else:
                fig_all_trends.add_hline(y=THRESHOLDS[s_config['threshold_key']]['normal_max'], line_dash="dot", line_color="lightgreen", row=i+1, col=1)
                fig_all_trends.add_hline(y=THRESHOLDS[s_config['threshold_key']]['warning_max'], line_dash="dash", line_color="orange", row=i+1, col=1)
                fig_all_trends.add_hline(y=THRESHOLDS[s_config['threshold_key']]['critical_max'], line_dash="dash", line_color="red", row=i+1, col=1)
            
            fig_all_trends.update_yaxes(title_text=s_config['unit'], row=i+1, col=1)

        fig_all_trends.update_layout(height=800, title_text="Live Trend Data Across All Brake Sensors", showlegend=False)
        st.plotly_chart(fig_all_trends, use_container_width=True)

    with tab_pressure:
        create_sensor_dashboard_section('brake_pressure', SENSOR_CONFIG['brake_pressure'], latest_data, df)

    with tab_temp:
        create_sensor_dashboard_section('brake_temperature', SENSOR_CONFIG['brake_temperature'], latest_data, df)

    with tab_pad_wear:
        create_sensor_dashboard_section('brake_pad_wear', SENSOR_CONFIG['brake_pad_wear'], latest_data, df)

    with tab_response_time:
        create_sensor_dashboard_section('brake_response_time', SENSOR_CONFIG['brake_response_time'], latest_data, df)

    with tab_predictive:
        st.header("🧠 Predictive Maintenance & Anomaly Detection")
        if st.session_state.latest_prediction:
            pred = st.session_state.latest_prediction
            col_pred1, col_pred2, col_pred3 = st.columns(3)
            
            pred_card_color = "lightgreen"
            if pred['condition'] == 'Warning':
                pred_card_color = "orange"
            elif pred['condition'] == 'Fault':
                pred_card_color = "red"

            with col_pred1:
                st.markdown(f"""
                <div style="background-color: {pred_card_color}; padding: 15px; border-radius: 10px; text-align: center; color: white;">
                    <h4>Predicted Brake Health</h4>
                    <h2>{pred['condition']}</h2>
                    <p>Confidence: {pred['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_pred2:
                st.info(f"""
                **Estimated Remaining Useful Life (RUL)**
                ## {pred['rul']} Hours
                """)
            
            with col_pred3:
                st.info(f"""
                **Anomaly Count**
                ## {st.session_state.anomaly_count}
                """)
            
            if show_ml_metrics:
                st.subheader("ML Model Performance")
                st.metric("Model Accuracy", f"{st.session_state.model_accuracy:.1f}%")
                st.markdown("*(Accuracy based on recent historical data)*")

        else:
            st.info("ML model predictions will appear here after sufficient data is collected and models are trained. Please wait or click 'TRAIN ML MODELS NOW'.")

    if show_raw_data:
        st.subheader("Raw Data Table (Last 20 readings)")
        st.dataframe(df.tail(20), use_container_width=True)


def create_dashboard():
    """
    Constructs the Streamlit web application dashboard.
    This function is called by Streamlit on every rerun.
    """
    st.set_page_config(page_title="Brake System Monitor", layout="wide")
    st.title("🛑 Advanced Brake System Monitoring & Testing Dashboard")

    # Initialize all session state variables if they don't exist
    initialize_session_state()

    # --- Start ML training thread if not already running ---
    if not st.session_state.ml_thread_running:
        logger.info("Starting ML training thread...")
        ml_thread = threading.Thread(target=ml_training_thread_func)
        ml_thread.daemon = True
        ml_thread.start()
        st.session_state.ml_thread_running = True

    # --- Ensure CSV file exists with headers immediately on startup ---
    if not os.path.exists(CONFIG['csv_file']):
        pd.DataFrame(columns=[
            'timestamp', 'brake_pressure', 'brake_temperature',
            'brake_pad_wear', 'brake_response_time', 'condition'
        ]).to_csv(CONFIG['csv_file'], index=False)
        st.info("Brake system data file created. Waiting for simulation data to accumulate in charts...")
        return

    # Auto-refresh functionality - generates data and refreshes display
    if st_autorefresh is not None:
        # Auto-refresh every 1000ms (1 second) - matches CONFIG['sim_interval']
        refresh_count = st_autorefresh(interval=CONFIG['sim_interval'] * 1000, key="brake_monitor_refresh")
        
        # Generate and save new data on each refresh
        generate_and_save_data()
    else:
        st.warning("⚠️ Auto-refresh not available. Install with: `pip install streamlit-autorefresh`")
        st.info("Manual refresh: Use the browser refresh button or add refresh controls.")
        
        # Fallback: Manual refresh button
        if st.button("🔄 Refresh Data", key="manual_refresh"):
            generate_and_save_data()

    # Sidebar controls for system interaction
    with st.sidebar:
        st.header("🔧 Brake System Control Panel")

        current_health = st.session_state.get('current_system_health', 'Normal')
        status_color_sidebar = "green"
        if current_health == 'Warning':
            status_color_sidebar = "orange"
        elif current_health == 'Fault':
            status_color_sidebar = "red"
        st.subheader("📊 System Status")
        st.markdown(f"**Health:** :{status_color_sidebar}[{current_health}]")
        st.markdown(f"**Mode:** {st.session_state.system_mode}")

        st.subheader("🚨 Emergency Controls")
        col1_eb, col2_rs = st.columns(2)

        with col1_eb:
            if st.button("🚨 EMERGENCY BRAKE", type="primary", key="emergency_brake_btn"):
                st.session_state.emergency_brake_active = True
                st.session_state.system_mode = 'Emergency'
                st.info("Emergency brake activated! Expect higher pressure and temperature readings.")
                
        with col2_rs:
            if st.button("🔄 RESET SYSTEM", type="secondary", key="reset_system_btn"):
                reset_system(st.session_state)
                st.success("System reset to normal!")
                st.session_state.fault_injection_enabled = True
        
        st.markdown("---")

        st.subheader("⚙️ Fault Injection & Simulation Mode")
        st.session_state.fault_injection_enabled = st.checkbox(
            "Enable Fault Injection", value=st.session_state.fault_injection_enabled,
            help="Toggle this to allow or prevent faults from being injected into the simulation."
        )

        fault_options = {
            'no_fault': '✅ No Faults',
            'fluid_leak': '💧 Fluid Leak (Low Pressure)',
            'overheating': '🔥 Overheating Brakes',
            'pad_wear': '⚠️ Worn Brake Pads',
            'delayed_response': '⏱️ Delayed Brake Response'
        }
        
        selected_fault = st.selectbox(
            "Select Fault Type to Inject",
            list(fault_options.keys()),
            format_func=lambda x: fault_options[x],
            disabled=not st.session_state.fault_injection_enabled,
            key="fault_type_selector"
        )
        
        fault_mode = "gradual"
        if selected_fault != 'no_fault':
            fault_mode = st.radio(
                "Fault Injection Mode",
                ["gradual", "sudden"],
                index=0,
                disabled=not st.session_state.fault_injection_enabled,
                key="fault_mode_selector"
            )

        if st.button("⚡ APPLY FAULT / NORMAL MODE", type="secondary", disabled=not st.session_state.fault_injection_enabled, key="apply_fault_btn"):
            if selected_fault == 'no_fault':
                reset_system(st.session_state)
                st.success("System reset to No Faults (Normal Conditions)!")
            else:
                inject_fault(selected_fault, fault_mode, st.session_state)
                if fault_mode == 'sudden':
                    st.success(f"Successfully injected a SUDDEN {fault_options[selected_fault]} fault!")
                else:
                    st.success(f"Successfully injected a GRADUAL {fault_options[selected_fault]} fault!")

        st.markdown("---")

        st.subheader("🤖 ML Model Training")
        if st.button("🏋️‍♀️ TRAIN ML MODELS NOW", key="train_ml_btn"):
            st.info("ML training initiated in the background. Check 'Predictive Analytics' tab for updates (models train every 20 seconds).")

        st.markdown("---")
        st.subheader("📊 Display Options")
        show_raw_data = st.checkbox("Show Raw Data Table", value=False, key="show_raw_data_checkbox")
        show_ml_metrics = st.checkbox("Show ML Metrics Details", value=True, key="show_ml_metrics_checkbox")

    # Render the main dashboard content
    render_dashboard_content(show_raw_data, show_ml_metrics)


if __name__ == "__main__":
    create_dashboard()