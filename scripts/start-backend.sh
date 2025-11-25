#!/bin/bash
set -e

echo "🐍 Starting Python backend services..."

cd "$(dirname "$0")/.."

# Start FastAPI server in background
echo "Starting FastAPI server on port 8000..."
cd backend
source .venv/bin/activate
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!
echo "FastAPI PID: $API_PID"

# Wait a bit for API server to start
sleep 3

# Start LiveKit agent
echo "Starting LiveKit agent..."
python -m agent.main &
AGENT_PID=$!
echo "Agent PID: $AGENT_PID"

echo ""
echo "✅ Backend services started!"
echo "FastAPI server: http://localhost:8000"
echo "WebSocket: ws://localhost:8000/ws/transcripts"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap "kill $API_PID $AGENT_PID 2>/dev/null" EXIT
wait
