import streamlit as st
import numpy as np
import pandas as pd
import time
from collections import deque
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Import for auto-refresh functionality
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# --- Enhanced Configuration Constants ---
SENSOR_METADATA = {
    # Engine System Sensors
    "Engine Oil Pressure": {"min": 30, "max": 70, "nominal": 50, "unit": "psi", "initial_rul": 10000, "rul_degrade": 5, "type": "pressure", "good_range_pct": 0.15, "critical_factor": 1.2, "category": "Pressure", "priority": "critical"},
    "Engine Coolant Temperature": {"min": 80, "max": 100, "nominal": 90, "unit": "°C", "initial_rul": 15000, "rul_degrade": 3, "type": "temperature", "good_range_pct": 0.12, "critical_factor": 1.5, "category": "Temperature", "priority": "critical"},
    "Engine RPM": {"min": 600, "max": 2000, "nominal": 1200, "unit": "RPM", "initial_rul": 20000, "rul_degrade": 7, "type": "general", "good_range_pct": 0.20, "critical_factor": 1.0, "category": "Performance", "priority": "high"},
    "Fuel Pressure": {"min": 40, "max": 60, "nominal": 50, "unit": "psi", "initial_rul": 12000, "rul_degrade": 4, "type": "pressure", "good_range_pct": 0.15, "critical_factor": 1.3, "category": "Pressure", "priority": "high"},
    "Engine Load": {"min": 20, "max": 90, "nominal": 55, "unit": "%", "initial_rul": 18000, "rul_degrade": 6, "type": "general", "good_range_pct": 0.25, "critical_factor": 1.1, "category": "Performance", "priority": "medium"},
    "Engine Manifold Pressure": {"min": 0.8, "max": 1.5, "nominal": 1.0, "unit": "bar", "initial_rul": 14000, "rul_degrade": 3, "type": "pressure", "good_range_pct": 0.12, "critical_factor": 1.2, "category": "Pressure", "priority": "medium"},

    # Braking System Sensors
    "Brake Pad Wear": {"min": 2, "max": 10, "nominal": 8, "unit": "mm", "initial_rul": 5000, "rul_degrade": 20, "type": "wear_inverse", "good_range_pct": 0.20, "critical_factor": 2.0, "category": "Wear", "priority": "critical"},
    "Brake Fluid Pressure": {"min": 1000, "max": 2000, "nominal": 1500, "unit": "psi", "initial_rul": 8000, "rul_degrade": 8, "type": "pressure", "good_range_pct": 0.15, "critical_factor": 1.4, "category": "Pressure", "priority": "critical"},
    "Brake Temperature": {"min": 50, "max": 200, "nominal": 100, "unit": "°C", "initial_rul": 10000, "rul_degrade": 10, "type": "temperature", "good_range_pct": 0.20, "critical_factor": 1.8, "category": "Temperature", "priority": "high"},
    "Brake Response Time": {"min": 0.1, "max": 0.8, "nominal": 0.3, "unit": "s", "initial_rul": 7000, "rul_degrade": 5, "type": "time", "good_range_pct": 0.20, "critical_factor": 1.6, "category": "Performance", "priority": "high"},
    "Brake Disc Thickness": {"min": 15, "max": 25, "nominal": 22, "unit": "mm", "initial_rul": 6000, "rul_degrade": 15, "type": "wear_inverse", "good_range_pct": 0.18, "critical_factor": 1.9, "category": "Wear", "priority": "critical"},
    "Brake Fluid Level": {"min": 20, "max": 100, "nominal": 80, "unit": "%", "initial_rul": 12000, "rul_degrade": 2, "type": "level", "good_range_pct": 0.15, "critical_factor": 1.5, "category": "Level", "priority": "medium"},

    # Tire System Sensors
    "Tire Pressure FL": {"min": 90, "max": 110, "nominal": 100, "unit": "psi", "initial_rul": 6000, "rul_degrade": 2, "type": "pressure", "good_range_pct": 0.08, "critical_factor": 1.2, "category": "Pressure", "priority": "critical"},
    "Tire Pressure FR": {"min": 90, "max": 110, "nominal": 100, "unit": "psi", "initial_rul": 6000, "rul_degrade": 2, "type": "pressure", "good_range_pct": 0.08, "critical_factor": 1.2, "category": "Pressure", "priority": "critical"},
    "Tire Pressure RL": {"min": 90, "max": 110, "nominal": 100, "unit": "psi", "initial_rul": 6000, "rul_degrade": 2, "type": "pressure", "good_range_pct": 0.08, "critical_factor": 1.2, "category": "Pressure", "priority": "critical"},
    "Tire Pressure RR": {"min": 90, "max": 110, "nominal": 100, "unit": "psi", "initial_rul": 6000, "rul_degrade": 2, "type": "pressure", "good_range_pct": 0.08, "critical_factor": 1.2, "category": "Pressure", "priority": "critical"},
    "Tire Temperature FL": {"min": 30, "max": 60, "nominal": 45, "unit": "°C", "initial_rul": 7000, "rul_degrade": 3, "type": "temperature", "good_range_pct": 0.12, "critical_factor": 1.3, "category": "Temperature", "priority": "high"},
    "Tire Temperature FR": {"min": 30, "max": 60, "nominal": 45, "unit": "°C", "initial_rul": 7000, "rul_degrade": 3, "type": "temperature", "good_range_pct": 0.12, "critical_factor": 1.3, "category": "Temperature", "priority": "high"},
    "Tire Temperature RL": {"min": 30, "max": 60, "nominal": 45, "unit": "°C", "initial_rul": 7000, "rul_degrade": 3, "type": "temperature", "good_range_pct": 0.12, "critical_factor": 1.3, "category": "Temperature", "priority": "high"},
    "Tire Temperature RR": {"min": 30, "max": 60, "nominal": 45, "unit": "°C", "initial_rul": 7000, "rul_degrade": 3, "type": "temperature", "good_range_pct": 0.12, "critical_factor": 1.3, "category": "Temperature", "priority": "high"},
    "Tire Tread Depth": {"min": 2, "max": 12, "nominal": 8, "unit": "mm", "initial_rul": 8000, "rul_degrade": 50, "type": "wear_inverse", "good_range_pct": 0.20, "critical_factor": 2.5, "category": "Wear", "priority": "critical"},

    # Vibration Monitoring System Sensors
    "Vibration Intensity Engine": {"min": 0.1, "max": 0.8, "nominal": 0.3, "unit": "g", "initial_rul": 9000, "rul_degrade": 6, "type": "vibration", "good_range_pct": 0.20, "critical_factor": 1.7, "category": "Vibration", "priority": "high"},
    "Vibration Intensity Chassis": {"min": 0.05, "max": 0.5, "nominal": 0.2, "unit": "g", "initial_rul": 11000, "rul_degrade": 4, "type": "vibration", "good_range_pct": 0.20, "critical_factor": 1.5, "category": "Vibration", "priority": "high"},
    "Vibration Intensity Cabin": {"min": 0.02, "max": 0.3, "nominal": 0.1, "unit": "g", "initial_rul": 13000, "rul_degrade": 3, "type": "vibration", "good_range_pct": 0.20, "critical_factor": 1.4, "category": "Vibration", "priority": "medium"},
    "Vibration Frequency": {"min": 10, "max": 100, "nominal": 50, "unit": "Hz", "initial_rul": 10000, "rul_degrade": 5, "type": "frequency", "good_range_pct": 0.20, "critical_factor": 1.3, "category": "Frequency", "priority": "high"},
    "Structural Stress": {"min": 100, "max": 800, "nominal": 400, "unit": "MPa", "initial_rul": 12000, "rul_degrade": 8, "type": "stress", "good_range_pct": 0.18, "critical_factor": 1.6, "category": "Stress", "priority": "high"},
}

# Enhanced component structure with key sensors
COMPONENTS = {
    "Engine System": {
        "key_sensors": ["Engine Oil Pressure", "Engine Coolant Temperature", "Engine RPM"],
        "all_sensors": [
            "Engine Oil Pressure", "Engine Coolant Temperature", "Engine RPM", 
            "Fuel Pressure", "Engine Load", 
            "Engine Manifold Pressure"
        ]
    },
    "Braking System": {
        "key_sensors": ["Brake Pad Wear", "Brake Fluid Pressure", "Brake Temperature"],
        "all_sensors": [
            "Brake Pad Wear", "Brake Fluid Pressure", "Brake Temperature", 
            "Brake Response Time", "Brake Disc Thickness", "Brake Fluid Level"
        ]
    },
    "Tire System": {
        "key_sensors": ["Tire Pressure FL", "Tire Temperature FL", "Tire Tread Depth"],
        "all_sensors": [
            "Tire Pressure FL", "Tire Pressure FR", "Tire Pressure RL", "Tire Pressure RR",
            "Tire Temperature FL", "Tire Temperature FR", "Tire Temperature RL", "Tire Temperature RR",
            "Tire Tread Depth"
        ]
    },
    "Vibration Monitoring System": {
        "key_sensors": ["Vibration Intensity Engine", "Vibration Frequency", "Structural Stress"],
        "all_sensors": [
            "Vibration Intensity Engine", "Vibration Intensity Chassis",
            "Vibration Intensity Cabin", "Vibration Frequency", "Structural Stress"
        ]
    }
}

# Parameter categories for organized display
PARAMETER_CATEGORIES = {
    "Pressure": ["psi", "bar", "MPa"],
    "Temperature": ["°C", "°F"],
    "Performance": ["RPM", "%", "s"],
    "Wear": ["mm"],
    "Vibration": ["g"],
    "Frequency": ["Hz"],
    "Stress": ["MPa"],
    "Electrical": ["V", "A"],
    "Level": ["%"]
}

HISTORY_LENGTH = 500
UPDATE_INTERVAL_SECONDS = 1.5

# --- Session State Management ---
def initialize_session_state():
    """Initialize all necessary session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.current_sensor_values = {}
        st.session_state.historical_data = {}
        st.session_state.health_statuses = {}
        st.session_state.anomaly_statuses = {}
        st.session_state.rul_predictions = {}
        st.session_state.alerts = []
        st.session_state.last_update_time = datetime.now()
        st.session_state.auto_update_enabled = True
        st.session_state.operation_mode = "Normal"
        st.session_state.fault_injection = False
        st.session_state.total_operating_hours = 0
        st.session_state.maintenance_recommendations = []
        st.session_state.component_health_scores = {}
        st.session_state.performance_metrics = {}
        st.session_state.trend_analysis = {}
        # Initialize selected_component with the first key, if components exist
        st.session_state.selected_component = list(COMPONENTS.keys())[0] if COMPONENTS else None
        st.session_state.initialized = True
        
        # Initialize data structures
        # Iterate over all_sensors in all components
        for component_data in COMPONENTS.values():
            for sensor_name in component_data["all_sensors"]:
                # Check if sensor exists in SENSOR_METADATA before initializing
                if sensor_name in SENSOR_METADATA:
                    st.session_state.historical_data[sensor_name] = deque(maxlen=HISTORY_LENGTH)
                    st.session_state.rul_predictions[sensor_name] = SENSOR_METADATA[sensor_name]["initial_rul"]
                else:
                    st.warning(f"Sensor '{sensor_name}' listed in COMPONENTS but not found in SENSOR_METADATA. Skipping initialization for this sensor.")


def simulate_operating_conditions():
    """Simulate different operating conditions that affect sensor readings"""
    conditions = ["Normal", "Heavy Load", "Extreme Weather", "Maintenance Due"]
    weights = [0.7, 0.15, 0.1, 0.05]
    return np.random.choice(conditions, p=weights)

def generate_correlated_sensor_values():
    """Generate sensor values with realistic correlations between components"""
    base_condition = simulate_operating_conditions()
    st.session_state.operation_mode = base_condition
    
    # Define condition multipliers
    condition_multipliers = {
        "Normal": 1.0,
        "Heavy Load": 1.3,
        "Extreme Weather": 1.2,
        "Maintenance Due": 1.5
    }
    
    multiplier = condition_multipliers.get(base_condition, 1.0)
    new_values = {}
    
    # Only generate values for sensors present in SENSOR_METADATA
    for sensor_name, metadata in SENSOR_METADATA.items():
        base_value = generate_single_sensor_value(sensor_name, multiplier)
        
        # Add cross-component correlations
        if "Engine" in sensor_name and base_condition == "Heavy Load":
            base_value *= 1.1
        elif "Brake" in sensor_name and base_condition == "Heavy Load":
            base_value *= 1.2
        elif "Tire" in sensor_name and base_condition == "Extreme Weather":
            base_value *= 1.15
        elif "Vibration" in sensor_name and base_condition == "Maintenance Due":
            base_value *= 1.25
            
        new_values[sensor_name] = base_value
    
    return new_values

def generate_single_sensor_value(sensor_name, condition_multiplier=1.0):
    """Generate realistic sensor values with degradation and noise"""
    metadata = SENSOR_METADATA.get(sensor_name)
    if not metadata:
        return np.random.rand() * 100 # Fallback if metadata is missing

    nominal = metadata["nominal"]
    min_val = metadata["min"]
    max_val = metadata["max"]
    value_type = metadata.get("type", "general")
    
    # Calculate degradation based on RUL
    current_rul = st.session_state.rul_predictions.get(sensor_name, metadata["initial_rul"])
    degradation_factor = 1 - (current_rul / metadata["initial_rul"])
    
    # Base noise
    range_span = max_val - min_val
    noise = np.random.normal(0, range_span * 0.02)
    
    if value_type == "wear_inverse":
        # For wear sensors, value decreases as RUL decreases (e.g., tread depth)
        degraded_value = max_val - ((max_val - min_val) * degradation_factor * 0.8)
        value = degraded_value + noise
    else:
        # For other sensors, add degradation noise
        degradation_noise = np.random.normal(0, range_span * 0.05 * degradation_factor)
        value = nominal + noise + degradation_noise
        
        # Apply condition multiplier
        if value_type in ["temperature", "pressure", "vibration"]:
            value *= condition_multiplier
    
    # Apply fault injection if enabled
    if st.session_state.fault_injection:
        fault_severity = np.random.uniform(1.2, 2.0)
        if np.random.random() < 0.3:  # 30% chance of fault
            value *= fault_severity
    
    return max(min_val, min(max_val, round(value, 2)))

def update_all_sensor_data():
    """Update all sensor data with enhanced simulation"""
    new_values = generate_correlated_sensor_values()
    
    for sensor_name, current_value in new_values.items():
        # Only process sensors that have been initialized in historical_data
        if sensor_name in st.session_state.historical_data:
            # Update current values and history
            st.session_state.current_sensor_values[sensor_name] = current_value
            st.session_state.historical_data[sensor_name].append(current_value)
            
            # Update RUL with enhanced degradation
            metadata = SENSOR_METADATA[sensor_name]
            base_degrade = metadata["rul_degrade"] / 10000
            
            # Accelerate degradation based on conditions
            condition_factor = 1.0
            if st.session_state.operation_mode == "Heavy Load":
                condition_factor = 1.5
            elif st.session_state.operation_mode == "Extreme Weather":
                condition_factor = 1.3
            elif st.session_state.operation_mode == "Maintenance Due":
                condition_factor = 2.0
                
            health_status = get_health_status(sensor_name, current_value)
            health_factor = {"Good": 1.0, "Degrading": 1.5, "Critical": 3.0}.get(health_status, 1.0)
            
            total_degrade = base_degrade * condition_factor * health_factor
            st.session_state.rul_predictions[sensor_name] = max(0, 
                st.session_state.rul_predictions[sensor_name] - total_degrade)
            
            # Update health and anomaly status
            st.session_state.health_statuses[sensor_name] = health_status
            st.session_state.anomaly_statuses[sensor_name] = detect_anomaly(sensor_name, current_value)
            
            # Generate alerts for critical conditions
            generate_alerts(sensor_name, current_value, health_status)
        else:
            # This should ideally not happen if initialization is correct
            # but serves as a safeguard for uninitialized sensors.
            st.warning(f"Skipping update for uninitialized sensor: {sensor_name}")
            
    # Update component health scores
    update_component_health_scores()
    
    # Update performance metrics
    update_performance_metrics()
    
    # Increment operating hours
    st.session_state.total_operating_hours += UPDATE_INTERVAL_SECONDS / 3600

def get_health_status(sensor_name, current_value):
    """Enhanced health status determination with fuzzy logic"""
    metadata = SENSOR_METADATA.get(sensor_name)
    if not metadata:
        return "Good"

    min_val = metadata["min"]
    max_val = metadata["max"]
    nominal = metadata["nominal"]
    value_type = metadata.get("type", "general")
    good_range_pct = metadata.get("good_range_pct", 0.1)
    
    if value_type == "wear_inverse":
        # For wear sensors, value decreases as RUL decreases (e.g., tread depth)
        # Lower values are worse, min_val is critical
        critical_threshold = min_val + (max_val - min_val) * 0.1 # 10% above min is critical
        degrading_threshold = min_val + (max_val - min_val) * 0.3 # 30% above min is degrading
        
        if current_value <= critical_threshold:
            return "Critical"
        elif current_value <= degrading_threshold:
            return "Degrading"
        else:
            return "Good"
    else:
        # For other sensors, values far from nominal are worse
        nominal_range_tolerance = (max_val - min_val) * good_range_pct
        
        # Define warning and critical bands around nominal
        warning_band_outer_lower = nominal - nominal_range_tolerance * 2
        warning_band_outer_upper = nominal + nominal_range_tolerance * 2
        
        critical_band_outer_lower = nominal - nominal_range_tolerance * 3
        critical_band_outer_upper = nominal + nominal_range_tolerance * 3

        # Ensure bands don't exceed min/max bounds
        warning_band_outer_lower = max(min_val, warning_band_outer_lower)
        warning_band_outer_upper = min(max_val, warning_band_outer_upper)
        critical_band_outer_lower = max(min_val, critical_band_outer_lower)
        critical_band_outer_upper = min(max_val, critical_band_outer_upper)
        
        if (current_value < critical_band_outer_lower or 
            current_value > critical_band_outer_upper):
            return "Critical"
        elif (current_value < warning_band_outer_lower or 
              current_value > warning_band_outer_upper):
            return "Degrading"
        else:
            return "Good"

def detect_anomaly(sensor_name, current_value):
    """Enhanced anomaly detection with multiple methods"""
    # Ensure the sensor has historical data initialized
    if sensor_name not in st.session_state.historical_data:
        return False

    history = list(st.session_state.historical_data[sensor_name])
    if len(history) < 30:
        return False
    
    # Statistical anomaly detection (Z-score)
    recent_history = history[-30:]
    mean = np.mean(recent_history)
    std_dev = np.std(recent_history)
    
    if std_dev < 0.01: # Avoid division by zero or very small std_dev
        return False
    
    z_score = abs(current_value - mean) / std_dev
    statistical_anomaly = z_score > 2.5 # Value is 2.5 standard deviations away from mean
    
    # Trend-based anomaly detection (simple change detection)
    if len(history) >= 10:
        # Calculate recent trend as the average difference between last 5 and previous 5 values
        recent_avg = np.mean(history[-5:])
        previous_avg = np.mean(history[-10:-5])
        
        # Check if there's a significant sudden change
        trend_change = abs(recent_avg - previous_avg)
        
        # Threshold for significant change, relative to sensor range or recent std dev
        metadata = SENSOR_METADATA.get(sensor_name)
        range_span = metadata["max"] - metadata["min"] if metadata else 1.0
        
        # A change is an anomaly if it's large relative to the sensor's typical fluctuation
        trend_anomaly = trend_change > (range_span * 0.05) and trend_change > (std_dev * 1.5)
    else:
        trend_anomaly = False
    
    return statistical_anomaly or trend_anomaly

def generate_alerts(sensor_name, current_value, health_status):
    """Generate contextual alerts for critical conditions"""
    # Check if this exact alert already exists to prevent duplicates
    # This is a simple check, can be made more sophisticated (e.g., check timestamp range)
    for alert in st.session_state.alerts:
        if alert["sensor"] == sensor_name and alert["severity"] == health_status and \
           (datetime.now() - alert["timestamp"]) < timedelta(minutes=5): # Avoid re-alerting too quickly
            return

    if health_status == "Critical":
        alert = {
            "timestamp": datetime.now(),
            "sensor": sensor_name,
            "severity": "Critical",
            "message": f"Critical: {sensor_name} at {current_value} {SENSOR_METADATA[sensor_name]['unit']}. Immediate action recommended.",
            "recommendation": get_maintenance_recommendation(sensor_name, health_status)
        }
        st.session_state.alerts.append(alert)
    elif health_status == "Degrading" and np.random.random() < 0.15:  # Reduced chance for degrading alerts
        alert = {
            "timestamp": datetime.now(),
            "sensor": sensor_name,
            "severity": "Warning",
            "message": f"Warning: {sensor_name} is degrading: {current_value} {SENSOR_METADATA[sensor_name]['unit']}. Monitor closely.",
            "recommendation": get_maintenance_recommendation(sensor_name, health_status)
        }
        st.session_state.alerts.append(alert)
    
    # Keep only last 20 alerts
    if len(st.session_state.alerts) > 20:
        st.session_state.alerts = st.session_state.alerts[-20:]

def get_maintenance_recommendation(sensor_name, health_status):
    """Generate maintenance recommendations based on sensor and health status"""
    recommendations = {
        "Engine Oil Pressure": {
            "Critical": "Immediate oil change and pressure system inspection required.",
            "Degrading": "Schedule oil change and filter replacement within 50 operating hours."
        },
        "Engine Coolant Temperature": {
            "Critical": "Check cooling system, radiator, and coolant level immediately. Engine overheating risk.",
            "Degrading": "Inspect cooling system for efficiency. Consider coolant flush."
        },
        "Engine RPM": {
            "Critical": "Investigate engine control unit (ECU) or fuel delivery system. Severe performance issue.",
            "Degrading": "Monitor engine performance. Check for irregular idle or acceleration."
        },
        "Fuel Pressure": {
            "Critical": "Inspect fuel lines, pump, and filter for blockage or leaks. Engine performance severely impacted.",
            "Degrading": "Check fuel filter and lines for signs of clogging or minor leaks."
        },
        "Engine Load": {
            "Critical": "Evaluate engine's power output and operational limits. Potential internal damage.",
            "Degrading": "Review operating practices. Ensure load is within design specifications."
        },
        "Engine Manifold Pressure": {
            "Critical": "Check for turbocharger/supercharger issues, manifold leaks, or sensor malfunction.",
            "Degrading": "Inspect intake system for minor leaks or restrictions."
        },
        "Brake Pad Wear": {
            "Critical": "Replace brake pads immediately - vehicle unsafe to operate.",
            "Degrading": "Schedule brake pad replacement within 100 operating hours."
        },
        "Brake Fluid Pressure": {
            "Critical": "Urgent brake system inspection for fluid leaks or master cylinder failure. Do not operate.",
            "Degrading": "Check brake fluid level and inspect for minor leaks. Bleed brake lines if necessary."
        },
        "Brake Temperature": {
            "Critical": "Allow brakes to cool. Inspect for seized calipers or dragging pads. Fire risk.",
            "Degrading": "Reduce heavy braking. Inspect for minor caliper issues or worn components."
        },
        "Brake Response Time": {
            "Critical": "Immediate inspection of brake lines, master cylinder, and ABS system. Critical safety hazard.",
            "Degrading": "Check brake lines for air or fluid contamination. Consider bleeding the system."
        },
        "Brake Disc Thickness": {
            "Critical": "Replace brake discs immediately. Risk of structural failure during braking.",
            "Degrading": "Plan for brake disc replacement in the near future."
        },
        "Brake Fluid Level": {
            "Critical": "Refill brake fluid and check for leaks. Critical safety hazard.",
            "Degrading": "Top up brake fluid. Inspect for gradual leaks over time."
        },
        "Tire Pressure FL": {
            "Critical": "Check for leaks and inflate tire immediately. Do not operate on low pressure.",
            "Degrading": "Monitor pressure regularly and inspect for slow leaks."
        },
        "Tire Pressure FR": {
            "Critical": "Check for leaks and inflate tire immediately. Do not operate on low pressure.",
            "Degrading": "Monitor pressure regularly and inspect for slow leaks."
        },
        "Tire Pressure RL": {
            "Critical": "Check for leaks and inflate tire immediately. Do not operate on low pressure.",
            "Degrading": "Monitor pressure regularly and inspect for slow leaks."
        },
        "Tire Pressure RR": {
            "Critical": "Check for leaks and inflate tire immediately. Do not operate on low pressure.",
            "Degrading": "Monitor pressure regularly and inspect for slow leaks."
        },
        "Tire Temperature FL": {
            "Critical": "Stop operation, allow tire to cool. Check for over-inflation, overloading, or brake dragging.",
            "Degrading": "Monitor tire temperature. Adjust load or driving style as needed."
        },
        "Tire Temperature FR": {
            "Critical": "Stop operation, allow tire to cool. Check for over-inflation, overloading, or brake dragging.",
            "Degrading": "Monitor tire temperature. Adjust load or driving style as needed."
        },
        "Tire Temperature RL": {
            "Critical": "Stop operation, allow tire to cool. Check for over-inflation, overloading, or brake dragging.",
            "Degrading": "Monitor tire temperature. Adjust load or driving style as needed."
        },
        "Tire Temperature RR": {
            "Critical": "Stop operation, allow tire to cool. Check for over-inflation, overloading, or brake dragging.",
            "Degrading": "Monitor tire temperature. Adjust load or driving style as needed."
        },
        "Tire Tread Depth": {
            "Critical": "Replace tire immediately for safety and legal compliance.",
            "Degrading": "Plan for tire replacement. Rotate tires if uneven wear."
        },
        "Vibration Intensity Engine": {
            "Critical": "Immediate investigation of engine mounts, crankshaft, or rotating components. High risk of failure.",
            "Degrading": "Perform engine balancing check. Inspect engine mounts."
        },
        "Vibration Intensity Chassis": {
            "Critical": "Inspect chassis, suspension, and wheel bearings for damage or looseness. Urgent attention needed.",
            "Degrading": "Check chassis components and wheel alignment for minor issues."
        },
        "Vibration Intensity Cabin": {
            "Critical": "Review cabin mounting, seating, and ergonomic setup. May indicate deeper structural issues.",
            "Degrading": "Assess cabin comfort and sound insulation. May be an early indicator."
        },
        "Vibration Frequency": {
            "Critical": "Identify source of specific frequency. Could indicate resonant failure or bearing issues.",
            "Degrading": "Analyze vibration spectrum for emerging patterns."
        },
        "Structural Stress": {
            "Critical": "Immediate inspection for cracks, deformities, or fatigue in structural components. Risk of catastrophic failure.",
            "Degrading": "Monitor stress points. Consider reinforcing or redesigning highly stressed areas."
        },
    }
    
    default_rec = {
        "Critical": f"Immediate inspection and maintenance required for {sensor_name}.",
        "Degrading": f"Schedule preventive maintenance for {sensor_name} soon."
    }
    
    return recommendations.get(sensor_name, default_rec).get(health_status, "Monitor closely for changes.")

def update_component_health_scores():
    """Calculate overall health scores for each component"""
    for component_name, component_data in COMPONENTS.items():
        health_scores = []
        for sensor_name in component_data["all_sensors"]:
            # Only consider sensors that exist in SENSOR_METADATA
            if sensor_name in SENSOR_METADATA:
                health_status = st.session_state.health_statuses.get(sensor_name, "Good")
                score = {"Good": 100, "Degrading": 60, "Critical": 20}.get(health_status, 100)
                
                # Weight by RUL
                current_rul = st.session_state.rul_predictions.get(sensor_name, 1000)
                initial_rul = SENSOR_METADATA[sensor_name]["initial_rul"]
                rul_factor = max(0.0, current_rul / initial_rul) # RUL factor goes from 0 to 1
                
                # Incorporate priority: critical sensors have a larger impact on component score
                priority_factor = {"critical": 2.0, "high": 1.5, "medium": 1.0, "low": 0.8}.get(SENSOR_METADATA[sensor_name].get("priority", "medium"), 1.0)

                weighted_score = (score * rul_factor) * priority_factor
                health_scores.append(weighted_score)
        
        # Normalize weighted scores to a 0-100 scale for the component
        if health_scores:
            # Recalculate total_priority_weight only for existing sensors
            total_priority_weight = sum([{"critical": 2.0, "high": 1.5, "medium": 1.0, "low": 0.8}.get(SENSOR_METADATA[s].get("priority", "medium"), 1.0) for s in component_data["all_sensors"] if s in SENSOR_METADATA])
            
            # Ensure division by zero is avoided if total_priority_weight is 0
            normalized_score = (sum(health_scores) / total_priority_weight) * (100 / 100) if total_priority_weight > 0 else 100
            st.session_state.component_health_scores[component_name] = max(0, min(100, normalized_score))
        else:
            st.session_state.component_health_scores[component_name] = 100 # Default if no sensors


def update_performance_metrics():
    """Update overall performance metrics"""
    all_health_scores = list(st.session_state.component_health_scores.values())
    
    st.session_state.performance_metrics = {
        "Overall Health": np.mean(all_health_scores) if all_health_scores else 100,
        "Critical Sensors": len([s for s in st.session_state.health_statuses.values() if s == "Critical"]),
        "Degrading Sensors": len([s for s in st.session_state.health_statuses.values() if s == "Degrading"]),
        "Anomalies Detected": len([a for a in st.session_state.anomaly_statuses.values() if a]),
        "Operating Hours": st.session_state.total_operating_hours,
        "Active Alerts": len([a for a in st.session_state.alerts if a["severity"] == "Critical"])
    }

# --- Enhanced Hierarchical Display Functions ---
def display_hierarchical_sensor_monitoring(selected_component):
    """Display hierarchical sensor monitoring: Component -> Key Sensors -> Parameters"""
    st.header("Hierarchical Sensor Monitoring")
    
    component_data = COMPONENTS[selected_component]
    
    # Display component overview
    display_component_overview(selected_component, component_data)
    
    # Key Sensors Section
    st.subheader(f"Key Sensors - {selected_component}")
    
    # Filter key_sensors to only include those present in SENSOR_METADATA
    valid_key_sensors = [s for s in component_data["key_sensors"] if s in SENSOR_METADATA]

    if valid_key_sensors:
        # Create tabs for key sensors
        key_sensor_tab_names = [f"{sensor}" for sensor in valid_key_sensors]
        key_sensor_tabs = st.tabs(key_sensor_tab_names)
        
        for i, sensor_name in enumerate(valid_key_sensors):
            with key_sensor_tabs[i]:
                display_key_sensor_analysis(sensor_name)
    else:
        st.info(f"No key sensors defined or available for {selected_component}.")
    
    st.markdown("---")
    
    # All Sensors by Parameter Category
    # Filter all_sensors to only include those present in SENSOR_METADATA
    valid_all_sensors = [s for s in component_data["all_sensors"] if s in SENSOR_METADATA]
    display_parameter_category_analysis(valid_all_sensors, selected_component)

def display_component_overview(component_name, component_data):
    """Display component overview with key metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Filter all_sensors to only include those present in SENSOR_METADATA for counts
    valid_all_sensors = [s for s in component_data["all_sensors"] if s in SENSOR_METADATA]
    valid_key_sensors = [s for s in component_data["key_sensors"] if s in SENSOR_METADATA]

    with col1:
        health_score = st.session_state.component_health_scores.get(component_name, 100)
        health_color = "green" if health_score >= 80 else ("orange" if health_score >= 50 else "red")
        st.markdown(f"**Component Health**")
        st.markdown(f"<h3 style='color:{health_color}'>{health_score:.1f}%</h3>", unsafe_allow_html=True)
    
    with col2:
        critical_count = len([s for s in valid_all_sensors 
                              if st.session_state.health_statuses.get(s) == "Critical"])
        st.metric(f"Critical Issues", critical_count) 
    
    with col3:
        degrading_count = len([s for s in valid_all_sensors 
                               if st.session_state.health_statuses.get(s) == "Degrading"])
        st.metric(f"Degrading Sensors", degrading_count)
    
    with col4:
        total_sensors = len(valid_all_sensors)
        key_sensors = len(valid_key_sensors)
        st.metric(f"Sensors (Key/Total)", f"{key_sensors}/{total_sensors}")

def display_key_sensor_analysis(sensor_name):
    """Display detailed analysis for a key sensor"""
    metadata = SENSOR_METADATA.get(sensor_name)
    if not metadata:
        st.error(f"Metadata not found for sensor: {sensor_name}")
        return

    current_value = st.session_state.current_sensor_values.get(sensor_name, "N/A")
    health_status = st.session_state.health_statuses.get(sensor_name, "Unknown")
    rul_estimate = max(0, int(st.session_state.rul_predictions.get(sensor_name, 0)))
    is_anomaly = st.session_state.anomaly_statuses.get(sensor_name, False)

    st.markdown(f"#### {sensor_name}")
    
    col1, col2, col3 = st.columns([1, 1, 2])

    # Column 1: Current Value & Status
    with col1:
        st.metric(
            label=f"Current Value ({sensor_name})", 
            value=f"{current_value} {metadata['unit']}" if current_value != "N/A" else "N/A",
        )
        health_color = "green" if health_status == "Good" else ("orange" if health_status == "Degrading" else "red")
        st.markdown(f"**Health Status:** <span style='color:{health_color}'>**{health_status}**</span>", unsafe_allow_html=True)
        st.markdown(f"**RUL:** {rul_estimate} cycles")

    # Column 2: Expected Range & Anomaly
    with col2:
        st.markdown("**Expected Operating Range:**")
        st.write(f"- Min: {metadata['min']} {metadata['unit']}")
        st.write(f"- Nominal: {metadata['nominal']} {metadata['unit']}")
        st.write(f"- Max: {metadata['max']} {metadata['unit']}")
        anomaly_color = "red" if is_anomaly else "green"
        st.markdown(f"**Anomaly Detected:** <span style='color:{anomaly_color}'>**{'Yes' if is_anomaly else 'No'}**</span>", unsafe_allow_html=True)

    # Column 3: Real-time Graph
    with col3:
        history_data = list(st.session_state.historical_data.get(sensor_name, deque()))
        if history_data:
            df_sensor_history = pd.DataFrame({
                'Time': [datetime.now() - timedelta(seconds=(len(history_data) - 1 - i) * UPDATE_INTERVAL_SECONDS) for i in range(len(history_data))],
                'Value': history_data
            })
            
            fig = px.line(df_sensor_history, x='Time', y='Value', 
                          title=f"{sensor_name} Trend",
                          labels={'Value': f"{sensor_name} ({metadata['unit']})"})
            
            # Add range lines
            fig.add_hline(y=metadata['min'], line_dash="dot", line_color="blue", annotation_text="Min", annotation_position="bottom right")
            fig.add_hline(y=metadata['max'], line_dash="dot", line_color="blue", annotation_text="Max", annotation_position="top right")
            fig.add_hline(y=metadata['nominal'], line_dash="dash", line_color="gray", annotation_text="Nominal", annotation_position="top left")

            # Highlight current value if critical or degrading
            line_color = "green"
            if health_status == "Degrading":
                line_color = "orange"
            elif health_status == "Critical":
                line_color = "red"
            fig.update_traces(line_color=line_color, line_width=2)

            fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"trend_chart_{sensor_name}_{st.session_state.last_update_time.timestamp()}")
        else:
            st.info("No historical data available yet.")
    st.markdown("---") # Separator after each key sensor

def display_parameter_category_analysis(sensor_list, component_name): # Added component_name for keying
    """Display all sensors within a component, grouped by parameter category."""
    
    # Group sensors by category
    sensors_by_category = {}
    for sensor_name in sensor_list:
        metadata = SENSOR_METADATA.get(sensor_name)
        if metadata:
            category = metadata.get("category", "Other")
            if category not in sensors_by_category:
                sensors_by_category[category] = []
            sensors_by_category[category].append(sensor_name)
    
    # Sort categories for consistent display
    sorted_categories = sorted(sensors_by_category.keys(), key=lambda x: list(PARAMETER_CATEGORIES.keys()).index(x) if x in PARAMETER_CATEGORIES else len(PARAMETER_CATEGORIES))

    for category in sorted_categories:
        st.markdown(f"### {category} Sensors")
        cols_per_row = 3
        
        # Iterate over sensors in the current category
        for i in range(0, len(sensors_by_category[category]), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, sensor_name in enumerate(sensors_by_category[category][i:i+cols_per_row]):
                with cols[j]:
                    metadata = SENSOR_METADATA.get(sensor_name)
                    if not metadata:
                        continue # Skip if metadata is missing

                    current_value = st.session_state.current_sensor_values.get(sensor_name, "N/A")
                    health_status = st.session_state.health_statuses.get(sensor_name, "Unknown")
                    
                    health_color = "green" if health_status == "Good" else ("orange" if health_status == "Degrading" else "red")

                    st.markdown(f"**{sensor_name}**")
                    st.markdown(f"<h4 style='color:{health_color}'>{current_value} {metadata['unit']}</h4>", unsafe_allow_html=True)
                    st.markdown(f"Status: <span style='color:{health_color}'>**{health_status}**</span>", unsafe_allow_html=True)
                    
                    # Mini-trend graph for each sensor
                    history_data = list(st.session_state.historical_data.get(sensor_name, deque()))
                    if history_data:
                        df_mini_trend = pd.DataFrame({
                            'Time': range(len(history_data)),
                            'Value': history_data
                        })
                        
                        fig_mini = px.line(df_mini_trend, x='Time', y='Value')
                        fig_mini.update_layout(
                            height=100, 
                            margin=dict(l=0, r=0, t=0, b=0), 
                            showlegend=False,
                            xaxis_visible=False,
                            yaxis_visible=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        fig_mini.update_traces(line_color=health_color, line_width=1)
                        # Add a subtle range for min/max on the mini-graph
                        fig_mini.add_hline(y=metadata['min'], line_dash="dot", line_color="lightgray", opacity=0.5, line_width=0.5)
                        fig_mini.add_hline(y=metadata['max'], line_dash="dot", line_color="lightgray", opacity=0.5, line_width=0.5)
                        st.plotly_chart(fig_mini, use_container_width=True, config={'displayModeBar': False}, key=f"mini_trend_{component_name}_{category}_{sensor_name}_{st.session_state.last_update_time.timestamp()}")
                    else:
                        st.text("No data.")
                    st.markdown("---") # Separator between individual sensors


# --- Main Dashboard Layout ---
st.set_page_config(layout="wide", page_title="Dump Truck Monitoring Dashboard")

initialize_session_state()

# Auto-update toggle in sidebar
st.sidebar.title("Dashboard Controls")
st.session_state.auto_update_enabled = st.sidebar.checkbox("Enable Auto-Update", value=st.session_state.auto_update_enabled, key="auto_update_checkbox")

# Manual fault injection toggle
st.session_state.fault_injection = st.sidebar.checkbox("Enable Fault Injection (for demo)", value=st.session_state.fault_injection, key="fault_injection_checkbox")

st.sidebar.subheader("Reset Data")
if st.sidebar.button("Reset All Data", key="reset_button"):
    st.session_state.clear()
    initialize_session_state()
    st.rerun() # Rerun to apply the reset

# Component selection (MOVED OUTSIDE THE LOOP)
# Ensure there's a default if COMPONENTS is empty, though it shouldn't be based on the provided structure.
default_component_index = 0
if st.session_state.selected_component in COMPONENTS:
    default_component_index = list(COMPONENTS.keys()).index(st.session_state.selected_component)

st.session_state.selected_component = st.selectbox(
    "Select Component", 
    list(COMPONENTS.keys()), 
    key="component_selector", # Unique key for this selectbox
    index=default_component_index
)

# --- Dashboard Navigation Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("🚛 System Dashboards")
st.sidebar.markdown("**Navigate to specialized dashboards:**")

# Dashboard URLs
DASHBOARD_URLS = {
    "🛑 Braking System": "http://localhost:8502",
    "🔧 Engine System": "http://localhost:8503", 
    "🛞 Tire System": "http://localhost:8504",
    "📊 Vibration Analysis": "http://localhost:8505"
}

# Create navigation buttons using st.link_button (opens in new tab)
for dashboard_name, url in DASHBOARD_URLS.items():
    st.sidebar.link_button(
        dashboard_name, 
        url, 
        use_container_width=True
    )

# Auto-refresh mechanism using streamlit-autorefresh
if st.session_state.auto_update_enabled and st_autorefresh:
    refresh_interval_ms = int(UPDATE_INTERVAL_SECONDS * 1000)  # Convert to milliseconds
    st_autorefresh(interval=refresh_interval_ms, key="main_autorefresh")

# Generate new data if auto-update is enabled and enough time has passed
current_time = datetime.now()
if (st.session_state.auto_update_enabled and 
    (current_time - st.session_state.last_update_time).total_seconds() >= UPDATE_INTERVAL_SECONDS):
    update_all_sensor_data()
    st.session_state.last_update_time = current_time

# Dashboard content (always displayed)
st.title("Dump Truck Fleet Monitoring")

col_overall1, col_overall2, col_overall3, col_overall4, col_overall5 = st.columns(5)

with col_overall1:
    overall_health_score = st.session_state.performance_metrics.get("Overall Health", 100)
    overall_health_color = "green" if overall_health_score >= 80 else ("orange" if overall_health_score >= 50 else "red")
    st.markdown(f"**Overall Fleet Health**")
    st.markdown(f"<h3 style='color:{overall_health_color}'>{overall_health_score:.1f}%</h3>", unsafe_allow_html=True)
    
with col_overall2:
    st.metric("Total Operating Hours", f"{st.session_state.performance_metrics.get('Operating Hours', 0):.1f} hrs")
with col_overall3:
    st.metric("Critical Alerts", st.session_state.performance_metrics.get("Active Alerts", 0))
with col_overall4:
    st.metric("Total Anomalies", st.session_state.performance_metrics.get("Anomalies Detected", 0))
with col_overall5:
    st.metric("Operating Mode", st.session_state.operation_mode)

st.markdown("---")

# Display Alerts
st.subheader("Active Alerts & Recommendations")
if st.session_state.alerts:
    alert_df = pd.DataFrame(st.session_state.alerts)
    alert_df["timestamp"] = alert_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(alert_df, use_container_width=True, height=200, key="alerts_dataframe")
else:
    st.info("No active alerts.", icon="ℹ️")
st.markdown("---")

# Hierarchical Sensor Monitoring Section - Pass the selected component
display_hierarchical_sensor_monitoring(st.session_state.selected_component)

# Show status message if auto-update is disabled
if not st.session_state.auto_update_enabled:
    st.info("Auto-update is disabled. Please enable it from the sidebar to start real-time monitoring.")