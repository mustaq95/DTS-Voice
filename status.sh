#!/bin/bash

# LiveKit Whisper Transcription System - Status Check Script

echo "========================================="
echo "📊 System Status Check"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

check_port() {
    local port=$1
    local name=$2
    if lsof -ti:$port > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $name (port $port): RUNNING"
        return 0
    else
        echo -e "${RED}✗${NC} $name (port $port): NOT RUNNING"
        return 1
    fi
}

check_process() {
    local pattern=$1
    local name=$2
    if pgrep -f "$pattern" > /dev/null 2>&1; then
        local pid=$(pgrep -f "$pattern" | head -1)
        echo -e "${GREEN}✓${NC} $name (PID: $pid): RUNNING"
        return 0
    else
        echo -e "${RED}✗${NC} $name: NOT RUNNING"
        return 1
    fi
}

# Check all services
echo "🔍 Checking Services..."
echo ""

check_port 7880 "LiveKit Server"
check_port 8000 "FastAPI Backend"
check_port 3000 "Next.js Frontend"
check_process "production_agent" "Production Agent"

echo ""
echo "📝 Recent Log Entries:"
echo ""

if [ -f /tmp/agent.log ]; then
    echo "Agent Log (last 3 lines):"
    tail -3 /tmp/agent.log 2>/dev/null | grep -v "DEBUG" || echo "  (no recent activity)"
    echo ""
fi

echo "========================================="
