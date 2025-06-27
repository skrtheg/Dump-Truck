# Dump Truck Fleet Monitoring System

> A comprehensive real-time monitoring system for dump truck fleets with ML-powered analytics, fault detection, and predictive maintenance capabilities.

## System Architecture

The system is currently implemented as a **Streamlit multipage application** with a central navigation hub that provides access to specialized monitoring dashboards for each truck subsystem.

#### Components:
- **`main.py`** - Central navigation hub and integrated fleet overview
- **`braking_system.py`** - Brake system monitoring (pressure, temperature, pad wear)
- **`tire_system.py`** - Tire system analysis (pressure, temperature, tread depth)
- **`engine_system.py`** - Engine performance monitoring
- **`vibration_system.py`** - Vibration and structural analysis

## Quick Start

### Prerequisites

- Python 3.12
- Required packages: `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `tensorflow`
- Docker

### Installation & Setup

1. **Setup Virtual Environment**:
   ```bash
   uv venv --python 3.12
   ```

2. **Install Dependencies**:
   ```bash
   uv sync 
   ```

## Running the System

#### Option 1: Central Navigation Hub (Recommended)
```bash
cd app
streamlit run main.py
```
This launches the central hub where you can navigate between all monitoring systems.

#### Option 2: Individual Component Access
```bash
cd app
streamlit run braking_system.py    # Brake monitoring
streamlit run tire_system.py       # Tire monitoring
streamlit run engine_system.py     # Engine monitoring
streamlit run vibration_system.py  # Vibration monitoring
```

#### Option 3: Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build individual container
docker build -t dump-truck-monitor .
docker run -p 8501:8501 dump-truck-monitor
```

#### Option 4: Quick Start Script
```bash
# Make script executable and run
chmod +x app/entrypoint.sh
./app/entrypoint.sh
```

## 🎯 Features

### Integrated Fleet Overview
- **Real-time system health monitoring**
- **Cross-system analytics and alerts**
- **Fleet-wide performance metrics**
- **Maintenance scheduling recommendations**

### Individual System Monitoring
Each subsystem provides:
- ✅ **Real-time sensor data visualization**
- 🤖 **ML-powered anomaly detection**
- 🚨 **Intelligent fault injection for testing**
- 📊 **Predictive maintenance analytics**
- 📈 **Historical data analysis**
- ⚙️ **Configurable thresholds and alerts**

### Advanced Analytics
- **Machine Learning Models**: SVM, Random Forest, Decision Trees, LSTM
- **Predictive Maintenance**: RUL (Remaining Useful Life) estimation
- **Anomaly Detection**: Statistical and trend-based algorithms
- **Health Scoring**: Component and fleet-wide health metrics
- **Interactive Controls**: Fault injection, system reset, ML training
- **Export Capabilities**: Data download in CSV format


