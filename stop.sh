#!/bin/bash

# LiveKit Whisper Transcription System - Stop Script
# This script cleanly stops all running services

echo "========================================="
echo "🛑 Stopping LiveKit Transcription System"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Kill production agent
echo "Stopping Production Agent..."
pkill -9 -f "production_agent" 2>/dev/null && print_status "Agent stopped" || print_warning "No agent running"

# Kill FastAPI
echo "Stopping FastAPI Backend (port 8000)..."
lsof -ti:8000 | xargs kill -9 2>/dev/null && print_status "FastAPI stopped" || print_warning "No FastAPI running"

# Kill Next.js
echo "Stopping Next.js Frontend (port 3000)..."
lsof -ti:3000 | xargs kill -9 2>/dev/null && print_status "Next.js stopped" || print_warning "No Next.js running"

# Kill LiveKit
echo "Stopping LiveKit Server (port 7880)..."
lsof -ti:7880 | xargs kill -9 2>/dev/null && print_status "LiveKit stopped" || print_warning "No LiveKit running"

# Kill any remaining Python multiprocessing workers
echo "Cleaning up any remaining worker processes..."
pkill -9 -f "multiprocessing.spawn" 2>/dev/null && print_status "Workers cleaned" || print_warning "No workers found"

echo ""
echo "========================================="
echo "✅ ALL SERVICES STOPPED"
echo "========================================="
echo ""
