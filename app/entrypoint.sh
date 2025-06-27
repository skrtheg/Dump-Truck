#!/bin/sh

echo "🚛 Starting Dump Truck Monitoring System..."
echo "=========================================="

# Define ports
MAIN_PORT=8501
BRAKING_PORT=8502
ENGINE_PORT=8503
TIRE_PORT=8504
VIBRATION_PORT=8505

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $port is already in use. Stopping existing process..."
        kill $(lsof -t -i:$port) 2>/dev/null || true
        sleep 2
    fi
}

# Function to start a dashboard
start_dashboard() {
    local name=$1
    local file=$2
    local port=$3
    
    echo "🚀 Starting $name on port $port..."
    check_port $port
    
    uv run streamlit run $file --server.port=$port --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false &
    
    # Store the PID for later cleanup
    echo $! >> .dashboard_pids
}

# Clean up any existing PID file
rm -f .dashboard_pids

echo ""
echo "Starting individual dashboards..."

# Start all dashboards
start_dashboard "Main Dashboard" "main.py" $MAIN_PORT
start_dashboard "Braking System" "braking_system.py" $BRAKING_PORT  
start_dashboard "Engine System" "engine_system.py" $ENGINE_PORT
start_dashboard "Tire System" "tire_system.py" $TIRE_PORT
start_dashboard "Vibration System" "vibration_system.py" $VIBRATION_PORT

echo ""
echo "⏳ Waiting for dashboards to start..."
sleep 8

echo ""
echo "✅ All dashboards started successfully!"
echo "=========================================="
echo "🌐 Access your dashboards at:"
echo "   🏠 Main Dashboard:     http://localhost:$MAIN_PORT"
echo "   🛑 Braking System:     http://localhost:$BRAKING_PORT"
echo "   🔧 Engine System:      http://localhost:$ENGINE_PORT" 
echo "   🛞 Tire System:        http://localhost:$TIRE_PORT"
echo "   📊 Vibration Analysis: http://localhost:$VIBRATION_PORT"
echo "=========================================="
echo ""
echo "💡 Open your browser and navigate to the Main Dashboard to access all systems"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping all dashboards..."
    
    if [ -f .dashboard_pids ]; then
        while read pid; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "   Stopping process $pid..."
                kill "$pid" 2>/dev/null || true
            fi
        done < .dashboard_pids
        rm -f .dashboard_pids
    fi
    
    # Also kill any remaining streamlit processes on our ports
    for port in $MAIN_PORT $BRAKING_PORT $ENGINE_PORT $TIRE_PORT $VIBRATION_PORT; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; then
            echo "   Cleaning up port $port..."
            kill $(lsof -t -i:$port) 2>/dev/null || true
        fi
    done
    
    echo "✅ All dashboards stopped successfully!"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Keep script running
echo "🔄 Dashboards are running... (Ctrl+C to stop)"
while true; do
    sleep 1
done