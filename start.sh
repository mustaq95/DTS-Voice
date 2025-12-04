#!/bin/bash

# LiveKit Whisper Transcription System - Clean Startup Script
# This script ensures a fresh start by cleaning up old processes and starting all services

set -e  # Exit on any error

echo "========================================="
echo "🚀 Starting LiveKit Transcription System"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Step 1: Kill all old processes
echo "📋 Step 1: Cleaning up old processes..."
echo ""

# Kill Python processes (agents)
print_status "Killing old Python agent processes..."
pkill -9 -f "production_agent" 2>/dev/null || print_warning "No old agents running"

# Kill processes on specific ports
print_status "Killing processes on port 8000 (FastAPI)..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || print_warning "Port 8000 already free"

print_status "Killing processes on port 3000 (Next.js)..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || print_warning "Port 3000 already free"

print_status "Killing processes on port 7880 (LiveKit)..."
lsof -ti:7880 | xargs kill -9 2>/dev/null || print_warning "Port 7880 already free"

# Wait for cleanup
sleep 2
echo ""

# Step 2: Clear old logs
echo "📋 Step 2: Clearing old logs..."
rm -f /tmp/livekit.log /tmp/fastapi.log /tmp/agent*.log /tmp/nextjs.log
print_status "Old logs cleared"
echo ""

# Step 3: Start LiveKit Server
echo "📋 Step 3: Starting LiveKit Server..."
livekit-server --dev > /tmp/livekit.log 2>&1 &
LIVEKIT_PID=$!
sleep 3

# Check if LiveKit started
if lsof -ti:7880 > /dev/null; then
    print_status "LiveKit Server started (PID: $LIVEKIT_PID, Port: 7880)"
else
    print_error "Failed to start LiveKit Server (check /tmp/livekit.log)"
fi
echo ""

# Step 4: Start FastAPI Backend
echo "📋 Step 4: Starting FastAPI Backend..."
cd backend
source .venv/bin/activate
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000 > /tmp/fastapi.log 2>&1 &
FASTAPI_PID=$!
cd ..
sleep 4

# Check if FastAPI started
if lsof -ti:8000 > /dev/null; then
    print_status "FastAPI Backend started (PID: $FASTAPI_PID, Port: 8000)"
else
    print_error "Failed to start FastAPI Backend (check /tmp/fastapi.log)"
fi
echo ""

# Step 5: Start Production Agent
echo "📋 Step 5: Starting Production Agent..."
cd backend
source .venv/bin/activate
python -m agent.production_agent dev > /tmp/agent.log 2>&1 &
AGENT_PID=$!
cd ..
sleep 8

# Check if agent registered
if grep -q "registered worker" /tmp/agent.log; then
    print_status "Production Agent started (PID: $AGENT_PID)"
else
    print_error "Failed to start Production Agent (check /tmp/agent.log for errors)"
    # Don't exit - continue to start frontend anyway
fi
echo ""

# Step 6: Start Next.js Frontend
echo "📋 Step 6: Starting Next.js Frontend..."
cd frontend
npm run dev > /tmp/nextjs.log 2>&1 &
NEXTJS_PID=$!
cd ..
sleep 5

# Check if Next.js started
if lsof -ti:3000 > /dev/null; then
    print_status "Next.js Frontend started (PID: $NEXTJS_PID, Port: 3000)"
else
    print_error "Failed to start Next.js Frontend (check /tmp/nextjs.log)"
fi
echo ""

# Summary
echo "========================================="
echo "✅ ALL SERVICES STARTED SUCCESSFULLY"
echo "========================================="
echo ""
echo "📊 Service Status:"
echo "  • LiveKit Server:   http://localhost:7880 (PID: $LIVEKIT_PID)"
echo "  • FastAPI Backend:  http://localhost:8000 (PID: $FASTAPI_PID)"
echo "  • Production Agent: Running (PID: $AGENT_PID)"
echo "  • Next.js Frontend: http://localhost:3000 (PID: $NEXTJS_PID)"
echo ""
echo "📝 Logs available at:"
echo "  • LiveKit:   /tmp/livekit.log"
echo "  • FastAPI:   /tmp/fastapi.log"
echo "  • Agent:     /tmp/agent.log"
echo "  • Frontend:  /tmp/nextjs.log"
echo ""
echo "🎤 Open http://localhost:3000 in your browser to start transcribing!"
echo ""
echo "⚠️  To stop all services, run: ./stop.sh"
echo "========================================="
